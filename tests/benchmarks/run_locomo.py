"""LoCoMo benchmark runner for memri.

LoCoMo (Long Conversation Memory) benchmark:
https://arxiv.org/abs/2401.12416

Dataset format (locomo10.json / locomo50.json):
  List of {sample_id, conversation, qa, ...}
  conversation: {speaker_a, speaker_b, session_N, session_N_date_time, ...}
  Each session_N: [{speaker, dia_id, text}, ...]
  qa: [{question, answer, evidence, category}, ...]

Category meanings:
  1 = Single-hop (recent)
  2 = Single-hop (episodic)
  3 = Multi-hop
  4 = Temporal reasoning
  5 = Open-domain / world knowledge

Usage:
    python -m tests.benchmarks.run_locomo --dataset path/to/locomo10.json
    python -m tests.benchmarks.run_locomo --dataset path/to/locomo10.json --max-sessions 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

from memri.config import MemriConfig
from memri.core.memory import MemriMemory


# ─────────────────────── Dataset loading ───────────────────────────────────

def _parse_conversations(conv: dict) -> list[dict]:
    """Convert the LoCoMo conversation dict into a flat list of {role, content} messages."""
    speaker_a = conv.get("speaker_a", "A")
    speaker_b = conv.get("speaker_b", "B")
    messages: list[dict] = []

    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )
    for sk in session_keys:
        date_key = f"{sk}_date_time"
        date_str = conv.get(date_key, "")
        if date_str:
            messages.append({"role": "system", "content": f"[Session: {date_str}]"})

        for turn in conv[sk]:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            role = "user" if speaker == speaker_a else "assistant"
            messages.append({"role": role, "content": f"{speaker}: {text}"})

    return messages


def load_locomo_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)

    sessions = []
    for item in raw:
        sessions.append({
            "session_id": item.get("sample_id", str(uuid.uuid4())),
            "conversations": _parse_conversations(item.get("conversation", {})),
            "questions": [
                {
                    "question": q["question"],
                    "answer": str(q["answer"]),
                    "category": q.get("category", 0),
                }
                for q in item.get("qa", [])
                if "answer" in q  # skip unanswerable adversarial items
            ],
        })
    return sessions


# ─────────────────────── Token-F1 scoring ──────────────────────────────────

def _token_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 (same as SQuAD metric)."""
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    pred_set = defaultdict(int)
    gold_set = defaultdict(int)
    for t in pred_tokens:
        pred_set[t] += 1
    for t in gold_tokens:
        gold_set[t] += 1
    common = sum(min(pred_set[t], gold_set[t]) for t in pred_set if t in gold_set)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_results(results: list[tuple[str, str, str, int]]) -> dict:
    """Return per-category and overall F1 scores."""
    by_cat: dict[int, list[float]] = defaultdict(list)
    all_f1: list[float] = []
    for _, pred, gold, cat in results:
        f1 = _token_f1(pred, gold)
        all_f1.append(f1)
        by_cat[cat].append(f1)

    scores = {
        "overall": sum(all_f1) / len(all_f1) if all_f1 else 0.0,
        "n": len(all_f1),
    }
    for cat, vals in sorted(by_cat.items()):
        scores[f"cat_{cat}"] = sum(vals) / len(vals)
    return scores


# ─────────────────────── Session runner ────────────────────────────────────

def _format_raw_context(session: dict) -> str:
    """Format the full conversation as plain text — no compression (Smriti-equivalent)."""
    lines = []
    for msg in session["conversations"]:
        role = msg["role"].upper()
        lines.append(f"[{role}] {msg['content']}")
    return "\n".join(lines)


async def run_session(
    memory: MemriMemory,
    session: dict,
    request_delay: float = 0.5,
    raw_mode: bool = False,
) -> list[tuple[str, str, str, int]]:
    """Ingest a session and answer its QA pairs.

    raw_mode=True:  pass the full conversation text directly (no Observer compression).
                    Equivalent to Smriti's approach — fair baseline comparison.
    raw_mode=False: use memri's Observer pipeline (tests compression quality).

    Returns list of (question, predicted, gold, category).
    """
    provider = memory.config.get_llm_provider()

    if raw_mode:
        context = _format_raw_context(session)
    else:
        thread_id = session["session_id"]
        memory.ensure_thread(thread_id, agent_type="benchmark")
        for msg in session["conversations"]:
            await memory.process_message(
                thread_id=thread_id,
                role=msg["role"],
                content=msg["content"],
            )
        context = memory.get_context(thread_id)

    results: list[tuple[str, str, str, int]] = []
    for qa in session["questions"]:
        question = qa["question"]
        gold = qa["answer"]
        category = qa.get("category", 0)

        if request_delay > 0:
            await asyncio.sleep(request_delay)

        prompt = (
            f"{context}\n\n"
            f"Question: {question}\n"
            "Answer briefly and directly. If you can't find it, say 'unknown'."
        )

        response = await provider.complete(
            system_prompt=(
                "You are answering questions about a past conversation. "
                "Answer concisely — one phrase or sentence. "
                "Base your answer only on the provided context."
            ),
            user_message=prompt,
            max_tokens=128,
        )
        results.append((question, response.content.strip(), gold, category))

    return results


# ─────────────────────── Main ───────────────────────────────────────────────

async def main(
    dataset_path: str,
    model: Optional[str] = None,
    max_sessions: int = 50,
    output_path: Optional[str] = None,
    request_delay: float = 0.5,
    raw_mode: bool = False,
):
    config = MemriConfig.load()
    if model:
        config.llm_model = model
        if model.startswith("gemini"):
            config.llm_provider = "gemini"
        elif model.startswith(("gpt-", "o1", "o3")):
            config.llm_provider = "openai"

    memory = MemriMemory(config)
    dataset = load_locomo_dataset(Path(dataset_path))[:max_sessions]

    mode_label = "RAW (full conversation, no compression)" if raw_mode else "MEMRI (observer pipeline)"
    print(f"Model: {config.llm_model} | Provider: {config.llm_provider}")
    print(f"Mode: {mode_label}")
    print(f"Sessions: {len(dataset)} | observe_threshold: {config.observe_threshold:,} tokens\n")

    all_results: list[tuple[str, str, str, int]] = []
    for i, session in enumerate(dataset):
        sid = session["session_id"]
        n_turns = len(session["conversations"])
        n_qa = len(session["questions"])
        print(f"[{i+1}/{len(dataset)}] {sid}  ({n_turns} turns, {n_qa} QA)")

        results = await run_session(memory, session, request_delay=request_delay, raw_mode=raw_mode)
        all_results.extend(results)

        running = score_results(all_results)
        print(f"  -> overall F1: {running['overall']:.3f}  ({running['n']} QA pairs so far)")

    final = score_results(all_results)
    print(f"\n{'='*60}")
    print(f"LoCoMo F1 Score: {final['overall']:.4f}  ({final['overall']*100:.1f}%)")
    print(f"Total QA pairs : {final['n']}")
    print(f"{'-'*40}")
    for k, v in sorted(final.items()):
        if k.startswith("cat_"):
            cat = k[4:]
            label = {
                "1": "Single-hop (recent)",
                "2": "Single-hop (episodic)",
                "3": "Multi-hop",
                "4": "Temporal reasoning",
                "5": "Open-domain",
            }.get(cat, f"Category {cat}")
            print(f"  Cat {cat} ({label}): {v:.3f}")
    print(f"{'='*60}\n")

    if output_path:
        with open(output_path, "w") as f:
            json.dump({"scores": final, "model": config.llm_model, "raw_mode": raw_mode}, f, indent=2)
        print(f"Results saved to {output_path}")

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to locomo10.json or locomo50.json")
    parser.add_argument("--model", default=None, help="Override model (e.g. gemini-2.5-flash)")
    parser.add_argument("--max-sessions", type=int, default=50)
    parser.add_argument("--output", default=None, help="Save scores JSON to this path")
    parser.add_argument("--request-delay", type=float, default=0.5,
                        help="Seconds to sleep between QA API calls (default 0.5, increase if hitting 429)")
    parser.add_argument("--raw", action="store_true",
                        help="Raw mode: pass full conversation directly (no Observer compression). "
                             "Use this for a fair comparison against full-context baselines.")
    args = parser.parse_args()
    asyncio.run(main(args.dataset, args.model, args.max_sessions, args.output, args.request_delay, args.raw))
