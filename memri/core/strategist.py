"""Strategist agent — ReasoningBank-style procedural memory for memri v0.2.

Distills generalizable reasoning strategies from conversation trajectories:
- After a session, analyzes what worked and what failed
- Extracts permanent "strategy" memories distinct from episodic observations
- Frustrated user messages trigger immediate high-priority strategy extraction

Storage: same SQLite store, observation_type='strategy' (vs 'observation')
Injection: strategies prepended before episodic observations in get_context()
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memri.storage.base import BaseMemoryStore
    from memri.llm.provider import BaseLLMProvider

# ── Frustration signals ───────────────────────────────────────────────────────
# Multi-layer detection: pattern matching catches clear signals fast;
# LLM fallback catches subtle tone that patterns miss.

_FRUSTRATION_PATTERNS = [
    # Explicit profanity / strong language
    r"\b(what the fuck|wtf|what the hell|what the heck|oh my god|omg|ffs|for fuck['s]* sake)\b",
    r"\b(shit|damn it|dammit|jesus christ|holy shit)\b",

    # References to prior instructions being ignored
    r"\b(i (already |just |literally )?(told|said|mentioned|asked|explained|showed|specified))\b",
    r"\b(i (said|mentioned) (this|that|it) before\b)",
    r"\b(as i (said|mentioned|noted|explained|pointed out))\b",
    r"\b(like i (said|told you|mentioned|asked))\b",
    r"\b(we (already |just )?(went over|discussed|talked about|covered) this)\b",

    # Repetition exasperation
    r"\b(how many times|again and again|over and over|repeatedly|keep (doing|making|forgetting|saying|telling))\b",
    r"\b(every (single )?(time|day)|always (doing|making|forgetting))\b",
    r"\b(still (doing|not|haven't|didn't|won't))\b",
    r"\b(again\?|seriously\?|really\?!|come on!|ugh|argh|aargh)\b",

    # Direct corrections / negations with emphasis
    r"\bno+,?\s+(no+,?\s+)*(that'?s? not|don'?t|stop|just)\b",
    r"\b(that'?s? (not what i|wrong|incorrect|not right|not what i (said|meant|asked|wanted)))\b",
    r"\b(not (like that|what i (want|need|asked|meant|said)))\b",
    r"\b(you('re| are) (not listening|missing the point|still|again))\b",

    # Imperatives expressing frustration
    r"\b(stop (doing|using|adding|making|always|it|that|this))\b",
    r"\b(never (do|use|add|make) (this|that|it) again\b)",
    r"\b(please (just |for once |stop |don'?t ))\b",
    r"\b(just (do|don't|stop|use|follow|listen|read))\b",

    # Rhetorical questions
    r"\b(why (do you|are you|would you|did you) (keep|always|still|again|even))\b",
    r"\b(why (can'?t you|won'?t you|don'?t you))\b",
    r"\b(how (hard|difficult) is (it|this))\b",
    r"\b(is (it|this) (really|that) (hard|difficult|complicated))\b",

    # Explicit frustration words
    r"\b(frustrated|frustrating|annoying|annoyed|irritating|irritated|fed up|tired of)\b",
    r"\b(disappointed|unacceptable|ridiculous|absurd)\b",

    # Dismissal / giving up
    r"\b(forget (it|this|that)|never ?mind|just drop (it|this))\b",
    r"\b(i give up|this is useless|this isn'?t working)\b",

    # ALL CAPS words (3+ chars = yelling signal, checked separately)
]

_FRUSTRATION_RE = re.compile("|".join(_FRUSTRATION_PATTERNS), re.IGNORECASE)

# ALL CAPS words of 3+ chars = frustration signal (e.g. "WHY", "STOP", "NO")
_CAPS_RE = re.compile(r'\b[A-Z]{3,}\b')


def is_frustrated(text: str) -> bool:
    """Return True if the message contains frustration signals."""
    if _FRUSTRATION_RE.search(text):
        return True
    # 2+ all-caps words = yelling
    caps_words = _CAPS_RE.findall(text)
    if len(caps_words) >= 2:
        return True
    # Repeated punctuation: ??? or !!! or ?!
    if re.search(r'[?!]{2,}', text):
        return True
    return False


# ── Prompts ───────────────────────────────────────────────────────────────────

_FRUSTRATION_EXTRACTION_PROMPT = """A user expressed frustration in a conversation. Extract the underlying rule or preference they were trying to enforce.

User message: {message}

Recent context (last few turns):
{context}

Extract ONE clear, actionable rule the agent violated or forgot.
Format: A single sentence starting with an imperative verb.
Examples:
- "Always check X before doing Y."
- "Never use Z pattern in this codebase."
- "When the user asks about X, always do Y first."

If no clear rule can be extracted, output: SKIP

Rule:"""

_SESSION_STRATEGY_PROMPT = """You are analyzing a completed coding/work session to extract generalizable reasoning strategies.

Session transcript (most recent {n_turns} turns):
{transcript}

Session outcome: {outcome}

Your task: Extract 1-3 generalizable strategies an AI assistant should remember when working with THIS user or on THIS type of task.

Rules for good strategies:
- Procedural: "When X happens, do Y" or "Always Z before W"
- Specific enough to act on, general enough to reuse
- Learned from what WORKED (outcome=success) or what FAILED (outcome=failure/frustration)
- NOT factual observations (those go in episodic memory)

Bad (too vague): "Be careful with code"
Bad (too specific): "Fix line 47 in auth.py"
Good: "When debugging auth errors in this project, check middleware order before inspecting token logic"
Good: "User prefers async/await over .then() chains — always use async patterns"
Good: "Before running any database migration, confirm the user has a backup — they lost data once before"

Output a JSON array of strategy strings. Output [] if nothing worth storing.
Example: ["Always verify X before Y", "When user mentions Z, check W first"]

Strategies:"""

_DEDUP_PROMPT = """Given existing strategies and a new candidate strategy, determine if the new one is a duplicate or adds new information.

Existing strategies:
{existing}

New candidate: {candidate}

Output ONLY one word: "keep" (adds new info) or "skip" (duplicate/redundant)."""


# ── Strategist ────────────────────────────────────────────────────────────────

class StrategistAgent:
    """Extracts and stores procedural strategies from session trajectories.

    Two modes:
    1. Immediate: called on every message — checks for frustration signals
    2. Post-session: called when session ends — distills strategies from full trajectory
    """

    def __init__(self, store: "BaseMemoryStore", provider: "BaseLLMProvider"):
        self.store = store
        self.provider = provider

    # ── Public API ────────────────────────────────────────────────────────────

    async def process_message(self, thread_id: str, role: str, content: str) -> None:
        """Check incoming message for frustration — extract strategy immediately if found."""
        if role != "user":
            return
        if not is_frustrated(content):
            return

        # Get last 6 turns as context
        messages = self.store.get_messages(thread_id, limit=6)
        context = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}" for m in messages[-6:]
        )

        try:
            response = await self.provider.complete(
                system_prompt="You extract concise, actionable rules from user frustration.",
                user_message=_FRUSTRATION_EXTRACTION_PROMPT.format(
                    message=content[:500],
                    context=context,
                ),
                max_tokens=128,
            )
            rule = response.content.strip()
            if rule and rule.upper() != "SKIP" and len(rule) > 10:
                await self._store_strategy(
                    thread_id=thread_id,
                    strategy=rule,
                    priority="critical",  # frustration = highest priority
                    source="frustration",
                )
        except Exception:
            pass  # never crash the main pipeline

    async def distill_session(
        self,
        thread_id: str,
        outcome: str = "unknown",
        max_turns: int = 40,
    ) -> list[str]:
        """Distill strategies from a completed session trajectory.

        Call this when a coding session ends or when the user explicitly
        marks something as done/failed.

        outcome: "success" | "failure" | "unknown"
        Returns list of new strategies stored.
        """
        messages = self.store.get_messages(thread_id, limit=max_turns)
        if len(messages) < 4:
            return []

        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content'][:400]}" for m in messages
        )

        try:
            response = await self.provider.complete(
                system_prompt="You extract generalizable reasoning strategies from agent trajectories.",
                user_message=_SESSION_STRATEGY_PROMPT.format(
                    n_turns=len(messages),
                    transcript=transcript[:8000],
                    outcome=outcome,
                ),
                max_tokens=512,
            )
            raw = response.content.strip()
            # Parse JSON array
            import json
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                return []
            strategies = json.loads(raw[start:end])
            if not isinstance(strategies, list):
                return []
        except Exception:
            return []

        stored = []
        for s in strategies[:3]:
            if isinstance(s, str) and len(s) > 15:
                if await self._is_new(thread_id, s):
                    priority = "high" if outcome == "success" else "medium"
                    await self._store_strategy(thread_id, s, priority, source=outcome)
                    stored.append(s)
        return stored

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _store_strategy(
        self,
        thread_id: str,
        strategy: str,
        priority: str,
        source: str,
    ) -> None:
        """Persist a strategy observation with priority emoji prefix."""
        emoji = {"critical": "🔴", "high": "🟡", "medium": "🔵"}.get(priority, "🔵")
        content = f"{emoji} [STRATEGY/{source.upper()}] {strategy}"
        self.store.add_observation(
            thread_id=thread_id,
            content=content,
            observation_type="strategy",
            token_count=len(content.split()),
        )

    async def _is_new(self, thread_id: str, candidate: str) -> bool:
        """Return True if candidate strategy is not a duplicate of existing ones."""
        existing = self.store.get_strategies(thread_id, limit=20)
        if not existing:
            return True
        existing_text = "\n".join(f"- {s['content']}" for s in existing[:10])
        try:
            response = await self.provider.complete(
                system_prompt="You detect duplicate strategies. Output only 'keep' or 'skip'.",
                user_message=_DEDUP_PROMPT.format(
                    existing=existing_text,
                    candidate=candidate,
                ),
                max_tokens=10,
            )
            return response.content.strip().lower().startswith("keep")
        except Exception:
            return True  # default keep on error
