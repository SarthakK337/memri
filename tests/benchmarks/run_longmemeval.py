"""LongMemEval-S benchmark runner for memri.

LongMemEval: https://arxiv.org/abs/2410.10813
Dataset: 500 QA pairs, each with ~50 sessions (~500 turns, ~115K tokens) haystack.

Scoring: LLM-as-judge (same methodology as official GPT-4o judge and Smriti).

Modes:
  --raw      Pass full conversation directly to model (no Observer compression).
             This is the ceiling — equivalent to Smriti's approach.
  (default)  Use memri Observer pipeline (tests compression quality).

Speed modes:
  --concurrency N   Run N examples in parallel (default 1 = sequential).
                    With Gemini paid tier, N=10 is safe. Free tier: N=2.
  --batch           Use Gemini Batch API (async, ~1-4h, no rate limits, 50% cheaper).
                    Only works with Gemini models. Best for full 500-example runs.

Target: >=80% accuracy (Smriti achieves ~80%+ with Cerebras Qwen-3 235B).

Usage:
    # Sequential (safe, slow ~40min):
    python -m tests.benchmarks.run_longmemeval --dataset path/to/longmemeval_s.json --raw

    # Parallel (fast ~5min with paid tier):
    python -m tests.benchmarks.run_longmemeval --dataset path/to/longmemeval_s.json --raw --concurrency 10

    # Batch API (no rate limits, async, submit and wait):
    python -m tests.benchmarks.run_longmemeval --dataset path/to/longmemeval_s.json --raw --batch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

from memri.config import MemriConfig
from memri.core.memory import MemriMemory


# ─────────────────────── Type-specific QA prompts ──────────────────────────
# Same prompt structure as Smriti — essential for accuracy on LongMemEval-S.

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
    """Format haystack_sessions into a single readable text block."""
    parts = []
    for i, (session, date) in enumerate(zip(sessions, dates)):
        lines = [f"[Session {i+1} | {date}]"]
        for turn in session:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            lines.append(f'{role}: {content}')
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────── Runner ────────────────────────────────────────

async def run_example(
    memory: MemriMemory,
    example: dict,
    raw_mode: bool = False,
    request_delay: float = 1.0,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> dict:
    """Run one LongMemEval example. Returns result dict."""
    async def _run():
        qid = example.get("question_id", str(uuid.uuid4()))
        question = example["question"]
        gold = example["answer"]
        question_type = example.get("question_type", "recall")
        question_date = example.get("question_date", "")
        sessions = example.get("haystack_sessions", [])
        dates = example.get("haystack_dates", [])

        provider = memory.config.get_llm_provider()

        if raw_mode:
            context = _sessions_to_text(sessions, dates)
        else:
            thread_id = f"lme-{qid}"
            memory.ensure_thread(thread_id, agent_type="benchmark-lme")
            for session_turns, date in zip(sessions, dates):
                for turn in session_turns:
                    await memory.process_message(
                        thread_id=thread_id,
                        role=turn.get("role", "user"),
                        content=f"[{date}] {turn.get('content', '')}",
                    )
            context = memory.get_context(thread_id)

        if request_delay > 0:
            await asyncio.sleep(request_delay)

        prompt = _get_prompt(question_type, context, question, question_date)
        qa_response = await provider.complete(
            system_prompt=(
                "You are answering questions about a user's conversation history with an AI assistant. "
                "Answer precisely from the provided context."
            ),
            user_message=prompt,
            max_tokens=256,
        )
        hypothesis = qa_response.content.strip()

        # LLM judge
        await asyncio.sleep(request_delay)
        judge_prompt = _JUDGE_PROMPT.format(
            question=question,
            reference=gold,
            hypothesis=hypothesis,
            question_type=question_type,
        )
        judge_response = await provider.complete(
            system_prompt="You are an answer evaluator. Output only 'correct' or 'incorrect'.",
            user_message=judge_prompt,
            max_tokens=128,
        )
        verdict = judge_response.content.strip().lower()
        if not verdict:
            verdict = "correct" if gold.lower() in hypothesis.lower() else "incorrect"
        correct = verdict.startswith("correct")

        return {
            "question_id": qid,
            "question_type": question_type,
            "question": question,
            "reference": gold,
            "hypothesis": hypothesis,
            "verdict": verdict,
            "correct": correct,
        }

    if semaphore:
        async with semaphore:
            return await _run()
    return await _run()


# ──────────────────────── Gemini Batch API mode ─────────────────────────────

async def run_batch_gemini(
    dataset: list[dict],
    model: str,
    api_key: str,
    raw_mode: bool,
    output_path: Optional[str],
) -> dict:
    """Submit all QA + judge calls as a single Gemini Batch job.

    Batch API: no rate limits, 50% cheaper, async (~1-4h for 500 examples).
    Polls every 60s until complete, then scores and saves results.
    """
    from google import genai
    from google.genai import types as gtypes

    client = genai.Client(api_key=api_key)

    qa_system = (
        "You are answering questions about a user's conversation history with an AI assistant. "
        "Answer precisely from the provided context."
    )
    judge_system = "You are an answer evaluator. Output only 'correct' or 'incorrect'."

    # Build all QA requests; we'll add judge requests after getting QA responses
    # Since batch doesn't support dependent requests, we do two batch rounds:
    # Round 1: all QA calls. Round 2: all judge calls (using round 1 results).

    print(f"Building batch requests for {len(dataset)} examples...")

    # Round 1: QA
    qa_requests = []
    for ex in dataset:
        qid = ex.get("question_id", str(uuid.uuid4()))
        question_type = ex.get("question_type", "recall")
        question_date = ex.get("question_date", "")
        sessions = ex.get("haystack_sessions", [])
        dates = ex.get("haystack_dates", [])
        context = _sessions_to_text(sessions, dates) if raw_mode else ""
        prompt = _get_prompt(question_type, context, ex["question"], question_date)

        qa_requests.append(gtypes.InlinedRequest(
            model=model,
            contents=prompt,
            metadata={"qid": qid},
            config=gtypes.GenerateContentConfig(
                system_instruction=qa_system,
                max_output_tokens=256,
            ),
        ))

    print(f"Submitting QA batch ({len(qa_requests)} requests)...")
    qa_job = await client.aio.batches.create(model=model, src=qa_requests)
    print(f"QA batch job: {qa_job.name} | state: {qa_job.state}")

    # Poll until done
    qa_job = await _poll_batch(client, qa_job.name)
    print(f"QA batch complete. Building judge requests...")

    # Extract QA responses (responses are in qa_job inline responses)
    qa_responses: dict[str, str] = {}
    for req, resp in zip(dataset, _batch_responses(qa_job)):
        qid = req.get("question_id", "")
        qa_responses[qid] = resp

    # Round 2: judge
    judge_requests = []
    for ex in dataset:
        qid = ex.get("question_id", "")
        question_type = ex.get("question_type", "recall")
        hypothesis = qa_responses.get(qid, "")
        judge_prompt = _JUDGE_PROMPT.format(
            question=ex["question"],
            reference=ex["answer"],
            hypothesis=hypothesis,
            question_type=question_type,
        )
        judge_requests.append(gtypes.InlinedRequest(
            model=model,
            contents=judge_prompt,
            metadata={"qid": qid},
            config=gtypes.GenerateContentConfig(
                system_instruction=judge_system,
                max_output_tokens=128,
            ),
        ))

    print(f"Submitting judge batch ({len(judge_requests)} requests)...")
    judge_job = await client.aio.batches.create(model=model, src=judge_requests)
    print(f"Judge batch job: {judge_job.name} | state: {judge_job.state}")
    judge_job = await _poll_batch(client, judge_job.name)
    print("Judge batch complete. Scoring...")

    judge_responses: dict[str, str] = {}
    for req, resp in zip(dataset, _batch_responses(judge_job)):
        qid = req.get("question_id", "")
        judge_responses[qid] = resp

    # Build results
    results = []
    for ex in dataset:
        qid = ex.get("question_id", "")
        hypothesis = qa_responses.get(qid, "")
        verdict = judge_responses.get(qid, "").strip().lower()
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

    scores = print_scores(results, model, "BATCH RAW" if raw_mode else "BATCH MEMRI")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"scores": scores, "model": model, "raw_mode": raw_mode,
                       "results": results}, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    return scores


async def _poll_batch(client, job_name: str, poll_interval: int = 60):
    """Poll a Gemini batch job until terminal state."""
    from google.genai.types import JobState
    terminal = {JobState.JOB_STATE_SUCCEEDED, JobState.JOB_STATE_FAILED,
                JobState.JOB_STATE_CANCELLED, JobState.JOB_STATE_PAUSED}
    while True:
        job = await client.aio.batches.get(name=job_name)
        print(f"  [{time.strftime('%H:%M:%S')}] Batch {job_name}: {job.state}")
        if job.state in terminal:
            if job.state != JobState.JOB_STATE_SUCCEEDED:
                raise RuntimeError(f"Batch job failed: {job.state} — {job.error}")
            return job
        await asyncio.sleep(poll_interval)


def _batch_responses(job) -> list[str]:
    """Extract text responses from a completed batch job's inline responses."""
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

    return {"overall": overall, "n": len(all_scores), "by_type": {k: sum(v)/len(v) for k, v in by_type.items()}}


# ─────────────────────────── Main ───────────────────────────────────────────

async def main(
    dataset_path: str,
    model: Optional[str] = None,
    max_examples: int = 500,
    output_path: Optional[str] = None,
    request_delay: float = 1.0,
    raw_mode: bool = False,
    checkpoint_path: Optional[str] = None,
    concurrency: int = 1,
    batch_mode: bool = False,
):
    config = MemriConfig.load()
    if model:
        config.llm_model = model
        if model.startswith("gemini"):
            config.llm_provider = "gemini"
        elif model.startswith(("gpt-", "o1", "o3")):
            config.llm_provider = "openai"

    dataset = load_dataset(Path(dataset_path))[:max_examples]

    # Batch mode — submit everything to Gemini Batch API and wait
    if batch_mode:
        if not config.llm_model.startswith("gemini"):
            raise ValueError("--batch only works with Gemini models")
        api_key = config.get_llm_provider().client._api_client._api_key
        return await run_batch_gemini(dataset, config.llm_model, api_key, raw_mode, output_path)

    memory = MemriMemory(config)
    mode_label = "RAW (full context, no compression)" if raw_mode else "MEMRI pipeline (observer)"
    print(f"Model       : {config.llm_model}")
    print(f"Mode        : {mode_label}")
    print(f"Examples    : {len(dataset)}")
    print(f"Concurrency : {concurrency}")
    print(f"Delay       : {request_delay}s between calls\n")

    # Load checkpoint
    done_results: list[dict] = []
    done_ids: set[str] = set()
    ckpt = Path(checkpoint_path) if checkpoint_path else Path(dataset_path).parent / "lme_checkpoint.json"
    if ckpt.exists():
        with open(ckpt, encoding="utf-8") as f:
            done_results = json.load(f)
        done_ids = {r["question_id"] for r in done_results}
        print(f"Checkpoint: {len(done_results)} done, {len(dataset) - len(done_ids)} remaining.")

    pending = [ex for ex in dataset if ex.get("question_id") not in done_ids]

    if concurrency > 1:
        # Parallel mode: process in batches of `concurrency`
        semaphore = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()

        async def _run_and_record(example: dict, idx: int):
            qid = example.get("question_id", "?")
            qtype = example.get("question_type", "?")
            try:
                result = await run_example(
                    memory, example, raw_mode=raw_mode,
                    request_delay=request_delay, semaphore=semaphore,
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
                        "question": example["question"], "reference": example["answer"],
                        "hypothesis": f"ERROR: {exc}", "verdict": "error", "correct": False,
                    })

        await asyncio.gather(*[_run_and_record(ex, i) for i, ex in enumerate(pending)])
    else:
        # Sequential mode
        for example in pending:
            qid = example.get("question_id", "?")
            qtype = example.get("question_type", "?")
            print(f"[{len(done_results)+1:>4}/{len(dataset)}] {qid}  ({qtype})", end=" ", flush=True)

            try:
                result = await run_example(memory, example, raw_mode=raw_mode, request_delay=request_delay)
                done_results.append(result)
                status = "OK" if result["correct"] else "WRONG"
                running_acc = sum(1 for r in done_results if r["correct"]) / len(done_results)
                print(f"-> {status}  |  running acc={running_acc:.1%}")
            except Exception as exc:
                print(f"-> ERROR: {exc}")
                done_results.append({
                    "question_id": qid, "question_type": qtype,
                    "question": example["question"], "reference": example["answer"],
                    "hypothesis": f"ERROR: {exc}", "verdict": "error", "correct": False,
                })

            with open(ckpt, "w", encoding="utf-8") as f:
                json.dump(done_results, f, ensure_ascii=False)

    scores = print_scores(done_results, config.llm_model, mode_label)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"scores": scores, "model": config.llm_model, "raw_mode": raw_mode,
                       "results": done_results}, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")

    ckpt.unlink(missing_ok=True)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to longmemeval_s.json")
    parser.add_argument("--model", default=None, help="Override model (e.g. gemini-2.5-flash)")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    parser.add_argument("--request-delay", type=float, default=1.0,
                        help="Seconds between API calls (default 1.0)")
    parser.add_argument("--raw", action="store_true",
                        help="Raw mode: pass full conversation directly (no Observer compression)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file path for resuming")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of parallel requests (default 1). Paid tier: 10 is safe.")
    parser.add_argument("--batch", action="store_true",
                        help="Use Gemini Batch API (async, no rate limits, 50%% cheaper). Gemini only.")
    args = parser.parse_args()
    asyncio.run(main(
        args.dataset, args.model, args.max_examples, args.output,
        args.request_delay, args.raw, args.checkpoint,
        args.concurrency, args.batch,
    ))
