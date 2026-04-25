"""LongMemEval-S benchmark runner for memri v1.0.

LongMemEval: https://arxiv.org/abs/2410.10813
Dataset: 500 QA pairs. Each example has its own independent haystack of ~50 sessions
(~115K tokens). The graph is built fresh per example, then queried.

Modes:
  --raw        Ceiling baseline: pass full ~115K token conversation directly to model.
  (default)    Graph mode: ingest sessions → graph search → answer from retrieved facts only.

Two-phase graph mode (recommended — separates slow ingestion from fast Q&A):

  Phase 1 — ingest + retrieve (LLM calls for fact extraction, cannot be batched):
    python -m tests.benchmarks.run_longmemeval \\
        --dataset longmemeval_s.json --ingest-only --contexts-output contexts.json

  Phase 2 — Q&A + judge using saved contexts (fully batchable, 50% cheaper):
    python -m tests.benchmarks.run_longmemeval \\
        --dataset longmemeval_s.json --from-contexts contexts.json \\
        --batch --output results_graph.json

  Or Phase 2 without batch (parallel):
    python -m tests.benchmarks.run_longmemeval \\
        --dataset longmemeval_s.json --from-contexts contexts.json \\
        --concurrency 10 --output results_graph.json

Raw baseline (single-phase, batchable):
    python -m tests.benchmarks.run_longmemeval \\
        --dataset longmemeval_s.json --raw --batch --output results_raw.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

from memri.config import MemriConfig


# ─────────────────────── Type-specific QA prompts ──────────────────────────

_QA_RECALL = """{context}

Instructions:
- Give a SHORT, precise answer (3-10 words). No full sentences unless needed.
- Use EXACT words from the sources above.
- For yes/no questions: output ONLY "Yes" or "No".
- If the information is genuinely not present: output "I don't know".

Question: {question}
Answer:"""

_QA_TEMPORAL = """{context}

Today's date: {question_date}

Instructions:
- Use the session dates above and today's date to compute any relative time (days/weeks/months ago).
- For "how many days/weeks/months ago": compute from the event date to today's date.
- For ordering questions: list events in chronological order based on session dates.
- Output ONLY the answer — a number + unit (e.g. "7 days", "3 weeks", "4 months") OR a date (e.g. "15 January 2023") OR an ordered list.
- Use digits: "3 days" not "three days".
- If genuinely not found: output "I don't know".

Question: {question}
Answer:"""

_QA_KNOWLEDGE_UPDATE = """{context}

Instructions:
- The user may have updated their information over time. Use the MOST RECENT value.
- Give a SHORT, precise answer (3-10 words).
- If the information has changed, use the LATEST value only.
- If genuinely not found: output "I don't know".

Question: {question}
Answer:"""

_QA_PREFERENCE = """{context}

Instructions:
- The question below is asking what the user would PREFER — describe their preferences, not a generic answer.
- Describe what the user would prefer based on their conversation history (2-4 sentences).
- Include what they would prefer AND what they would NOT prefer if mentioned.
- Use specific details from the conversation (brands, topics, styles, constraints).
- If genuinely not found: output "I don't know".

Question: {question}
Describe the user's preferences relevant to this question:"""


def _get_prompt(question_type: str, context: str, question: str, question_date: str = "") -> str:
    qt = question_type.lower()
    if "temporal" in qt:
        return _QA_TEMPORAL.format(context=context, question=question, question_date=question_date or "unknown")
    if "knowledge-update" in qt:
        return _QA_KNOWLEDGE_UPDATE.format(context=context, question=question)
    if "preference" in qt:
        return _QA_PREFERENCE.format(context=context, question=question)
    return _QA_RECALL.format(context=context, question=question)


_JUDGE_PROMPT = """You are evaluating whether a predicted answer is correct.

Question: {question}
Reference answer: {reference}
Predicted answer: {hypothesis}
Question type: {question_type}

Evaluation rules:
- single-session-user / single-session-assistant / multi-session:
  "correct" if the prediction contains the key information from the reference.
- temporal-reasoning:
  "correct" if the prediction gives the right date/duration. Allow:
    * off-by-one for day/week counts (reference says "7 days", prediction says "6 days" or "8 days" -> correct)
    * different date formats ("Jan 15 2023" = "15 January 2023" = "2023/01/15")
    * the reference may list multiple acceptable answers (e.g. "7 days. 8 days is also acceptable") — any of them counts
    * ordering questions: correct if the order of events matches the reference
- knowledge-update:
  "correct" if the prediction gives the MOST RECENT/UPDATED value (ignore older values).
- single-session-preference:
  "correct" if the prediction captures the key preference(s) from the reference, even partially.
  The reference is a long description — the prediction does NOT need to match it word for word.
  Focus on whether the core preference is present (e.g. brand, topic, style, constraint).

Output ONLY one word: "correct" or "incorrect"."""


# ────────────────────────── Dataset loading ────────────────────────────────

def _sessions_to_text(sessions: list, dates: list) -> str:
    """Format haystack_sessions into a single readable text block (raw mode)."""
    parts = []
    for i, (session, date) in enumerate(zip(sessions, dates)):
        lines = [f"[Session {i+1} | {date}]"]
        for turn in session:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            lines.append(f"{role}: {content}")
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────── Graph ingestion + retrieval ──────────────────────

def _format_graph_context(results: list) -> str:
    """Format graph search results as a compact context string for the LLM.

    Only the top-k retrieved facts are included — NOT the full graph.
    Typically 200-800 tokens vs 115K tokens in raw mode.
    """
    if not results:
        return "No relevant facts found in memory."
    lines = ["[Memory — retrieved facts ranked by relevance]"]
    for r in results:
        node = r.node
        date = node.temporal_date or node.session_date or ""
        suffix = f" [{date}]" if date else ""
        lines.append(f"- {node.content}{suffix}")
    return "\n".join(lines)


async def _with_retry(coro_fn, max_retries: int = 8, base_delay: float = 15.0):
    """Retry an async callable with exponential backoff.

    Retries on rate-limit (429), server errors (500/503), and transient network errors.
    Gives up after max_retries and re-raises the last exception.
    """
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except Exception as e:
            err = str(e).lower()
            retryable = any(x in err for x in (
                "429", "rate limit", "quota", "resource exhausted",
                "500", "502", "503", "service unavailable",
                "timeout", "connection", "reset", "overloaded",
            ))
            if not retryable or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)  # 10s, 20s, 40s, 80s, 160s
            print(f"    [retry {attempt+1}/{max_retries}] {e!r} — waiting {delay:.0f}s...")
            await asyncio.sleep(delay)


async def build_graph_context(example: dict, config: MemriConfig, graphs_dir: Path) -> str:
    """Phase 1: ingest all sessions into a persistent graph dir, search, return context.

    Each example gets its own directory under graphs_dir/{qid}/.
    Progress is saved after every session — if the process crashes mid-example,
    the next run resumes from the last completed session instead of starting over.
    The graph dir is deleted after the context is successfully extracted.
    """
    import shutil
    from memri.core.graph_memory import GraphMemoryEngine
    from memri.llm.graph_adapter import GraphLLMAdapter

    qid = example.get("question_id", "?")
    question = example["question"]
    sessions = example.get("haystack_sessions", [])
    dates = example.get("haystack_dates", [])

    graph_dir = graphs_dir / qid
    graph_dir.mkdir(parents=True, exist_ok=True)
    progress_file = graph_dir / "sessions_done.txt"

    # Load which sessions are already ingested
    done_sessions: set[int] = set()
    if progress_file.exists():
        for line in progress_file.read_text().splitlines():
            if line.strip().isdigit():
                done_sessions.add(int(line.strip()))

    provider = config.get_llm_provider()
    original_complete = provider.complete
    async def _complete_with_retry(*args, **kwargs):
        return await _with_retry(lambda: original_complete(*args, **kwargs))
    provider.complete = _complete_with_retry

    adapter = GraphLLMAdapter(provider, config.llm_model)
    engine = GraphMemoryEngine(graph_dir, adapter)  # loads existing graph if present

    n_sessions = len(sessions)
    remaining = n_sessions - len(done_sessions)
    if done_sessions:
        print(f"  [{qid}] resuming — {len(done_sessions)} done, {remaining} remaining", flush=True)
    else:
        print(f"  [{qid}] ingesting {n_sessions} sessions...", flush=True)

    for i, (session_turns, date) in enumerate(zip(sessions, dates)):
        if i in done_sessions:
            continue
        session_text = "\n".join(
            f"{t.get('role', 'user').capitalize()}: {t.get('content', '').strip()}"
            for t in session_turns
            if t.get("content", "").strip()
        )
        if session_text:
            date_str = str(date) if date is not None else None
            await _with_retry(lambda t=session_text, idx=i, d=date_str: engine.add(t, session_index=idx, session_date=d))

        # Save progress to disk immediately after each session
        with open(progress_file, "a") as f:
            f.write(f"{i}\n")
        done_sessions.add(i)
        print(f"  [{qid}] session {i+1}/{n_sessions} done", flush=True)

    # Retrieve relevant facts for this question
    print(f"  [{qid}] searching...", flush=True)
    results = await _with_retry(lambda: engine.search(question, top_k=15))
    context = _format_graph_context(results)

    # Close ChromaDB file handles before deleting (Windows file lock fix)
    engine.embeddings.close()

    # Delete the graph dir — context is now saved in checkpoint, graph not needed
    try:
        shutil.rmtree(graph_dir, ignore_errors=True)
    except Exception:
        pass

    return context


# ──────────────────────────── Single example runner ────────────────────────

async def run_example(
    example: dict,
    config: MemriConfig,
    raw_mode: bool = False,
    context_override: Optional[str] = None,
    request_delay: float = 1.0,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> dict:
    """Run one LongMemEval example end-to-end. Returns result dict.

    context_override: pre-built context string (from --from-contexts). If provided,
                      skips ingestion entirely and goes straight to Q&A.
    """
    async def _run():
        qid = example.get("question_id", str(uuid.uuid4()))
        question = example["question"]
        gold = example["answer"]
        question_type = example.get("question_type", "recall")
        question_date = example.get("question_date", "")
        sessions = example.get("haystack_sessions", [])
        dates = example.get("haystack_dates", [])

        provider = config.get_llm_provider()

        # Build context
        if context_override is not None:
            context = context_override                        # pre-built graph context
        elif raw_mode:
            context = _sessions_to_text(sessions, dates)     # full ~115K token dump
        else:
            context = await build_graph_context(example, config)  # graph: ingest + search

        if request_delay > 0:
            await asyncio.sleep(request_delay)

        # Q&A
        prompt = _get_prompt(question_type, context, question, question_date)
        qa_response = await _with_retry(lambda: provider.complete(
            system_prompt=(
                "You are answering questions about a user's conversation history with an AI assistant. "
                "Answer precisely from the provided context."
            ),
            user_message=prompt,
            max_tokens=256,
        ))
        hypothesis = qa_response.content.strip()

        # Judge
        await asyncio.sleep(request_delay)
        judge_prompt = _JUDGE_PROMPT.format(
            question=question,
            reference=gold,
            hypothesis=hypothesis,
            question_type=question_type,
        )
        judge_response = await _with_retry(lambda: provider.complete(
            system_prompt="You are an answer evaluator. Output only 'correct' or 'incorrect'.",
            user_message=judge_prompt,
            max_tokens=128,
        ))
        verdict = judge_response.content.strip().lower()
        if not verdict:
            verdict = "correct" if gold.lower() in hypothesis.lower() else "incorrect"

        return {
            "question_id": qid,
            "question_type": question_type,
            "question": question,
            "reference": gold,
            "hypothesis": hypothesis,
            "verdict": verdict,
            "correct": verdict.startswith("correct"),
        }

    if semaphore:
        async with semaphore:
            return await _run()
    return await _run()


# ──────────────────────── Phase 1: ingest-only ─────────────────────────────

async def run_ingest_only(
    dataset: list[dict],
    config: MemriConfig,
    contexts_output: str,
    concurrency: int = 3,
    checkpoint_path: Optional[str] = None,
    graphs_dir: Optional[Path] = None,
) -> None:
    """Phase 1: build graph context for each example, save to JSON.

    Output: [{question_id, question, answer, question_type, question_date, context}, ...]
    Resume-safe: saves a checkpoint after each example.
    """
    ckpt = Path(checkpoint_path) if checkpoint_path else Path(contexts_output).with_suffix(".ckpt.json")
    done: dict[str, dict] = {}
    if ckpt.exists():
        with open(ckpt, encoding="utf-8") as f:
            for row in json.load(f):
                done[row["question_id"]] = row
        print(f"Checkpoint: {len(done)} done, {len(dataset) - len(done)} remaining.")

    pending = [ex for ex in dataset if ex.get("question_id") not in done]
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    gdir = graphs_dir or Path(contexts_output).parent / "lme_graphs"
    gdir.mkdir(parents=True, exist_ok=True)

    async def _ingest_one(ex: dict):
        qid = ex.get("question_id", "?")
        try:
            async with semaphore:
                print(f"[START] {qid}", flush=True)
                ctx = await build_graph_context(ex, config, gdir)
            row = {
                "question_id": qid,
                "question": ex["question"],
                "answer": ex["answer"],
                "question_type": ex.get("question_type", "recall"),
                "question_date": ex.get("question_date", ""),
                "context": ctx,
            }
            async with lock:
                done[qid] = row
                total_done = len(done)
                print(f"[{total_done:>4}/{len(dataset)}] {qid} ingested ({len(ctx)} chars)")
                with open(ckpt, "w", encoding="utf-8") as f:
                    json.dump(list(done.values()), f, ensure_ascii=False)
        except Exception as e:
            print(f"  ERROR {qid}: {e}")

    await asyncio.gather(*[_ingest_one(ex) for ex in pending])

    # Write final output
    rows = list(done.values())
    with open(contexts_output, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nContexts saved to {contexts_output} ({len(rows)} examples)")
    ckpt.unlink(missing_ok=True)


# ──────────────────────── Gemini Batch API ─────────────────────────────────

async def run_batch_gemini(
    examples: list[dict],
    contexts: dict[str, str],
    model: str,
    api_key: str,
    raw_mode: bool,
    output_path: Optional[str],
) -> dict:
    """Submit all QA + judge calls as two Gemini Batch jobs.

    examples: dataset rows
    contexts: {question_id: context_string} — pre-built or raw
    """
    from google import genai
    from google.genai import types as gtypes

    client = genai.Client(api_key=api_key)
    qa_system = (
        "You are answering questions about a user's conversation history with an AI assistant. "
        "Answer precisely from the provided context."
    )
    judge_system = "You are an answer evaluator. Output only 'correct' or 'incorrect'."

    # Round 1: QA batch
    qa_requests = []
    for ex in examples:
        qid = ex.get("question_id", "")
        context = contexts.get(qid, "No context available.")
        prompt = _get_prompt(
            ex.get("question_type", "recall"),
            context,
            ex["question"],
            ex.get("question_date", ""),
        )
        qa_requests.append(gtypes.EmbedContentRequest(
            model=model,
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=qa_system,
                max_output_tokens=256,
            ),
        ))

    print(f"Submitting QA batch ({len(qa_requests)} requests)...")
    qa_job = await client.aio.batches.create(model=model, src=qa_requests)
    print(f"QA batch: {qa_job.name} | state: {qa_job.state}")
    qa_job = await _poll_batch(client, qa_job.name)

    qa_responses: dict[str, str] = {}
    for ex, resp_text in zip(examples, _batch_responses(qa_job)):
        qa_responses[ex.get("question_id", "")] = resp_text

    # Round 2: judge batch
    judge_requests = []
    for ex in examples:
        qid = ex.get("question_id", "")
        hypothesis = qa_responses.get(qid, "")
        judge_prompt = _JUDGE_PROMPT.format(
            question=ex["question"],
            reference=ex["answer"],
            hypothesis=hypothesis,
            question_type=ex.get("question_type", "recall"),
        )
        judge_requests.append(gtypes.EmbedContentRequest(
            model=model,
            contents=judge_prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=judge_system,
                max_output_tokens=128,
            ),
        ))

    print(f"Submitting judge batch ({len(judge_requests)} requests)...")
    judge_job = await client.aio.batches.create(model=model, src=judge_requests)
    print(f"Judge batch: {judge_job.name} | state: {judge_job.state}")
    judge_job = await _poll_batch(client, judge_job.name)

    results = []
    for ex, verdict_text in zip(examples, _batch_responses(judge_job)):
        qid = ex.get("question_id", "")
        hypothesis = qa_responses.get(qid, "")
        verdict = verdict_text.strip().lower()
        if not verdict:
            verdict = "correct" if ex["answer"].lower() in hypothesis.lower() else "incorrect"
        results.append({
            "question_id": qid,
            "question_type": ex.get("question_type", "recall"),
            "question": ex["question"],
            "reference": ex["answer"],
            "hypothesis": hypothesis,
            "verdict": verdict,
            "correct": verdict.startswith("correct"),
        })

    mode = "BATCH RAW" if raw_mode else "BATCH GRAPH"
    scores = print_scores(results, model, mode)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"scores": scores, "model": model, "mode": mode,
                       "results": results}, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    return scores


async def _poll_batch(client, job_name: str, poll_interval: int = 60):
    from google.genai.types import JobState
    terminal = {JobState.JOB_STATE_SUCCEEDED, JobState.JOB_STATE_FAILED,
                JobState.JOB_STATE_CANCELLED, JobState.JOB_STATE_PAUSED}
    while True:
        job = await client.aio.batches.get(name=job_name)
        print(f"  [{time.strftime('%H:%M:%S')}] {job_name}: {job.state}")
        if job.state in terminal:
            if job.state != JobState.JOB_STATE_SUCCEEDED:
                raise RuntimeError(f"Batch job failed: {job.state} — {job.error}")
            return job
        await asyncio.sleep(poll_interval)


def _batch_responses(job) -> list[str]:
    responses = []
    for resp in (job.responses or []):
        try:
            text = resp.response.candidates[0].content.parts[0].text or ""
        except Exception:
            text = ""
        responses.append(text.strip())
    return responses


# ────────────────────────── Scoring ─────────────────────────────────────────

def print_scores(results: list[dict], model: str, mode: str) -> dict:
    by_type: dict[str, list] = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(1 if r["correct"] else 0)

    all_scores = [s for scores in by_type.values() for s in scores]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

    print(f"\n{'='*58}")
    print(f"LongMemEval-S Results  ({mode})")
    print(f"Model: {model}")
    print(f"{'='*58}")
    for qt, scores in sorted(by_type.items()):
        acc = sum(scores) / len(scores) * 100
        print(f"  {qt:35s}: n={len(scores):3d}  acc={acc:.1f}%")
    print(f"  {'Overall':35s}: n={len(all_scores):3d}  acc={overall*100:.1f}%")
    print(f"{'='*58}\n")

    return {"overall": overall, "n": len(all_scores),
            "by_type": {k: sum(v)/len(v) for k, v in by_type.items()}}


# ─────────────────────────── Main ───────────────────────────────────────────

async def main(
    dataset_path: str,
    model: Optional[str] = None,
    max_examples: int = 500,
    output_path: Optional[str] = None,
    request_delay: float = 1.0,
    raw_mode: bool = False,
    checkpoint_path: Optional[str] = None,
    concurrency: int = 5,
    batch_mode: bool = False,
    ingest_only: bool = False,
    contexts_output: Optional[str] = None,
    from_contexts: Optional[str] = None,
):
    config = MemriConfig.load()
    if model:
        config.llm_model = model
        if model.startswith("gemini"):
            config.llm_provider = "gemini"
        elif model.startswith(("gpt-", "o1", "o3")):
            config.llm_provider = "openai"

    dataset = load_dataset(Path(dataset_path))[:max_examples]
    print(f"Model      : {config.llm_model}")
    print(f"Examples   : {len(dataset)}")

    # ── Phase 1: ingest-only ────────────────────────────────────────────────
    if ingest_only:
        out = contexts_output or str(Path(dataset_path).parent / "lme_contexts.json")
        print(f"Mode       : GRAPH INGEST ONLY → {out}")
        print(f"Concurrency: {concurrency}\n")
        await run_ingest_only(dataset, config, out, concurrency, checkpoint_path)
        return

    # ── Load pre-built contexts (Phase 2 or direct graph run) ───────────────
    contexts: dict[str, str] = {}
    if from_contexts:
        with open(from_contexts, encoding="utf-8") as f:
            for row in json.load(f):
                contexts[row["question_id"]] = row["context"]
        print(f"Contexts   : loaded {len(contexts)} from {from_contexts}")

    # ── Batch mode (raw or from-contexts) ───────────────────────────────────
    if batch_mode:
        if not config.llm_model.startswith("gemini"):
            raise ValueError("--batch only works with Gemini models")

        # Build contexts for raw mode (no LLM needed — just text formatting)
        if raw_mode and not contexts:
            contexts = {
                ex.get("question_id", ""): _sessions_to_text(
                    ex.get("haystack_sessions", []),
                    ex.get("haystack_dates", []),
                )
                for ex in dataset
            }

        if not contexts:
            raise ValueError(
                "--batch requires either --raw or --from-contexts.\n"
                "Run Phase 1 first: --ingest-only --contexts-output contexts.json"
            )

        api_key = config.get_llm_provider()._client.api_key
        return await run_batch_gemini(
            dataset, contexts, config.llm_model, api_key, raw_mode, output_path
        )

    # ── Sequential / concurrent mode ────────────────────────────────────────
    mode_label = (
        "RAW (full context)"   if raw_mode else
        "GRAPH (from contexts)" if from_contexts else
        "GRAPH (ingest + search)"
    )
    print(f"Mode       : {mode_label}")
    print(f"Concurrency: {concurrency}\n")

    # Checkpoint
    done_results: list[dict] = []
    done_ids: set[str] = set()
    ckpt = Path(checkpoint_path) if checkpoint_path else Path(dataset_path).parent / "lme_checkpoint.json"
    if ckpt.exists():
        with open(ckpt, encoding="utf-8") as f:
            done_results = json.load(f)
        done_ids = {r["question_id"] for r in done_results}
        print(f"Checkpoint : {len(done_results)} done, {len(dataset) - len(done_ids)} remaining.")

    pending = [ex for ex in dataset if ex.get("question_id") not in done_ids]
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def _run_one(ex: dict):
        qid = ex.get("question_id", "?")
        qtype = ex.get("question_type", "?")
        ctx = contexts.get(qid) if contexts else None
        try:
            result = await run_example(
                ex, config,
                raw_mode=raw_mode,
                context_override=ctx,
                request_delay=request_delay,
                semaphore=semaphore,
            )
            async with lock:
                done_results.append(result)
                status = "OK" if result["correct"] else "WRONG"
                running_acc = sum(1 for r in done_results if r["correct"]) / len(done_results)
                print(f"[{len(done_results):>4}/{len(dataset)}] {qid} ({qtype}) -> {status}  | acc={running_acc:.1%}")
                with open(ckpt, "w", encoding="utf-8") as f:
                    json.dump(done_results, f, ensure_ascii=False)
        except Exception as exc:
            async with lock:
                print(f"  ERROR {qid}: {exc}")
                done_results.append({
                    "question_id": qid, "question_type": qtype,
                    "question": ex["question"], "reference": ex["answer"],
                    "hypothesis": f"ERROR: {exc}", "verdict": "error", "correct": False,
                })

    await asyncio.gather(*[_run_one(ex) for ex in pending])

    scores = print_scores(done_results, config.llm_model, mode_label)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"scores": scores, "model": config.llm_model, "mode": mode_label,
                       "results": done_results}, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")

    ckpt.unlink(missing_ok=True)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongMemEval-S benchmark for memri v1.0")
    parser.add_argument("--dataset", required=True, help="Path to longmemeval_s.json")
    parser.add_argument("--model", default=None, help="Override model (e.g. gemini-2.5-flash)")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--output", default=None, help="Save final results JSON here")
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file path for resuming")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Parallel workers (default 5). Phase 1 ingestion: use 2-3.")
    # Mode flags
    parser.add_argument("--raw", action="store_true",
                        help="Raw mode: full conversation context, no graph (ceiling baseline)")
    parser.add_argument("--batch", action="store_true",
                        help="Use Gemini Batch API for Q&A+judge. Requires --raw or --from-contexts.")
    parser.add_argument("--ingest-only", action="store_true",
                        help="Phase 1: ingest sessions into graph, save retrieved contexts to JSON")
    parser.add_argument("--contexts-output", default=None,
                        help="Output path for --ingest-only (default: lme_contexts.json)")
    parser.add_argument("--from-contexts", default=None,
                        help="Phase 2: skip ingestion, load pre-built contexts from this JSON file")
    args = parser.parse_args()

    asyncio.run(main(
        dataset_path=args.dataset,
        model=args.model,
        max_examples=args.max_examples,
        output_path=args.output,
        request_delay=args.request_delay,
        raw_mode=args.raw,
        checkpoint_path=args.checkpoint,
        concurrency=args.concurrency,
        batch_mode=args.batch,
        ingest_only=args.ingest_only,
        contexts_output=args.contexts_output,
        from_contexts=args.from_contexts,
    ))
