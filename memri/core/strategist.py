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
# Three layers: word/phrase list, structural regex, punctuation/caps signals.

# Layer 1: individual words and short phrases that indicate frustration
# Checked via simple substring matching (fast, no regex overhead)
_FRUSTRATION_WORDS: frozenset[str] = frozenset([
    # Profanity / strong language
    "fuck", "fucking", "fucked", "wtf", "what the fuck", "what the hell",
    "what the heck", "ffs", "for fuck's sake", "for fucks sake",
    "shit", "bullshit", "horseshit", "holy shit", "oh shit",
    "damn", "dammit", "damn it", "goddamn", "goddammit",
    "hell", "bloody hell", "for hell's sake",
    "crap", "oh crap", "piss", "pissed", "pissed off",
    "bastard", "screw this", "screw that", "screw you",
    "smh", "omfg", "omg", "oh my god", "oh my goodness",
    "jesus christ", "jesus", "christ",

    # Exasperation / disbelief
    "seriously", "are you serious", "you're serious",
    "unbelievable", "i can't believe this", "i cannot believe this",
    "ridiculous", "absurd", "outrageous", "preposterous",
    "pathetic", "useless", "worthless", "pointless", "hopeless",
    "insane", "crazy", "nuts", "bonkers",
    "what is wrong with you", "something is wrong with you",
    "you must be kidding", "you've got to be kidding",
    "you gotta be kidding", "you're kidding me", "you're joking",
    "this is a joke", "this has to be a joke",
    "i can't deal with this", "i cannot deal with this",
    "i can't take this", "i cannot take this",
    "i can't do this anymore", "enough is enough",
    "this is too much", "i've had enough", "i have had enough",

    # Prior instruction references
    "i told you", "i already told you", "i just told you", "i literally told you",
    "i said", "i already said", "as i said", "like i said",
    "i mentioned", "i already mentioned", "as i mentioned", "like i mentioned",
    "i explained", "i already explained", "as i explained",
    "i showed you", "i already showed you", "i showed this",
    "i asked", "i already asked", "i specifically asked",
    "i specified", "i clearly specified", "i clearly stated",
    "i noted", "i pointed out", "i made it clear",
    "as i pointed out", "as i noted", "as i stated",
    "we already discussed", "we went over this", "we covered this",
    "we already went over", "we talked about this",
    "don't you remember", "do you not remember", "have you forgotten",
    "did you forget", "you forgot", "you seem to have forgotten",
    "i keep saying", "i keep telling you",

    # Repetition exasperation
    "how many times", "how many times do i have to",
    "how many times have i", "how many times must i",
    "again and again", "over and over", "time and time again",
    "repeatedly", "constantly", "continuously", "nonstop", "non-stop",
    "every single time", "every time i", "each and every time",
    "always doing", "always making", "always forgetting",
    "keep doing", "keep making", "keep forgetting",
    "keep repeating", "keep saying", "keep adding",
    "still doing", "still not", "still haven't", "still don't",
    "yet again", "once again", "here we go again", "here we are again",
    "not again", "oh not again",

    # Emphatic corrections
    "no no no", "no no", "nope nope",
    "absolutely not", "definitely not", "certainly not",
    "that's wrong", "completely wrong", "totally wrong", "all wrong",
    "that's incorrect", "completely incorrect", "totally incorrect",
    "that's not right", "that's not correct",
    "that's not what i said", "that's not what i meant",
    "that's not what i asked", "that's not what i wanted",
    "that's not what i need", "that's not what i requested",
    "not what i want", "not what i need", "not what i asked",
    "not what i meant", "not what i said", "not what i specified",
    "missed the point", "missing the point", "you're missing the point",
    "you missed the point", "you got it wrong", "you're wrong",
    "that's the opposite", "that's backwards",
    "you're not listening", "you are not listening",
    "you're not understanding", "you don't understand",
    "you clearly don't understand", "clearly you don't understand",
    "you clearly dont understand", "clearly you dont understand",

    # Imperatives of frustration
    "stop it", "stop doing this", "stop doing that",
    "stop adding", "stop using", "stop making",
    "quit it", "quit doing this", "cut it out",
    "enough already", "just stop", "please stop",
    "never do this again", "never do that again",
    "don't do this again", "don't do that again",
    "please just", "for once", "just once",
    "is it too much to ask",

    # Rhetorical questions
    "why would you", "why did you even", "why are you still",
    "why do you keep", "why do you always", "why do you still",
    "why can't you", "why won't you", "why don't you",
    "how hard is it", "how difficult is it",
    "is it really that hard", "is it that difficult",
    "is it too hard", "is it so hard",
    "what part of", "which part of",
    "do you even understand", "did you even read",
    "are you even reading", "are you even listening",
    "did you even look", "did you even check",
    "do you even know",

    # Explicit emotion words
    "frustrated", "frustration", "frustrating",
    "annoyed", "annoying", "annoyance",
    "irritated", "irritating", "irritation",
    "fed up", "sick of this", "sick and tired",
    "tired of this", "tired of you",
    "exasperated", "exasperation",
    "disappointed", "disappointing", "disappointment",
    "angry", "anger", "angered", "so angry",
    "upset", "very upset", "quite upset",
    "furious", "livid", "irate", "enraged",
    "pissed", "pissed off",
    "infuriated", "infuriating",
    "displeased", "not pleased", "not happy",
    "this is unacceptable", "unacceptable",
    "this is unprofessional", "this is inexcusable",

    # Dismissal / giving up
    "forget it", "forget this", "forget that",
    "nevermind", "never mind", "just forget it",
    "i give up", "giving up on this",
    "this is hopeless", "this is pointless",
    "this is useless", "this isn't working",
    "waste of time", "wasting my time",
    "not worth it", "not worth my time",
    "i'm done", "i am done", "done with this",
    "scrap everything", "start over",

    # Sarcasm signals
    "oh great", "just great", "that's just great",
    "oh perfect", "just perfect", "that's perfect",  # sarcastic
    "oh wonderful", "oh fantastic", "oh brilliant",
    "thanks for nothing", "great job",  # sarcastic
    "wow thanks", "oh wow",

    # Physical/emotional expressions
    "ugh", "uggh", "ughhh", "ugggh",
    "argh", "aargh", "aaarrgh", "aaargh",
    "grr", "grrr",
    "sigh", "*sigh*", "big sigh",
    "facepalm", "*facepalm*",

    # Emphasis / intensifiers in frustration context
    "i literally", "literally just told",
    "i clearly", "i obviously", "i specifically",
    "i explicitly told", "i explicitly said",
    "i repeatedly", "i have repeatedly",
])

# Pre-process to sorted list of phrases (longer first so multi-word matches first)
_FRUSTRATION_PHRASES: list[str] = sorted(_FRUSTRATION_WORDS, key=len, reverse=True)

# Layer 2: structural regex patterns for things the word list can't catch
_FRUSTRATION_PATTERNS = [
    # Repeated "no" (no no, no no no)
    r"\bno[\s,!]+no\b",
    # References to prior turns with any verb
    r"\bi (already |just |literally |clearly |specifically )?(told|said|mentioned|asked|explained|showed|specified|noted|stated|wrote|typed)\b",
    r"\b(as|like) i (said|mentioned|noted|explained|pointed out|specified|stated|asked)\b",
    # How many times variations
    r"\bhow many times\b",
    # Repetition references
    r"\b(again and again|over and over|time and time again|every single time)\b",
    r"\b(keep (doing|making|adding|using|forgetting|repeating|saying|telling|ignoring))\b",
    r"\b(you('re| are) (not listening|missing the point|still doing|doing it again|ignoring))\b",
    # Rhetorical questions
    r"\bwhy (do you|are you|would you|did you|can'?t you|won'?t you|don'?t you) (keep|always|still|even|just)\b",
    r"\b(how (hard|difficult|complicated) is (it|this|that))\b",
    r"\b(is (it|this|that) (really |so |that )?(hard|difficult|complicated|much to ask))\b",
    r"\b(what part of .{1,40} don'?t you (understand|get))\b",
    r"\b(did you (even |actually )?(read|look|check|see|understand))\b",
    # Imperatives with emphasis
    r"\b(stop (doing|using|adding|making|always|it|that|this|ignoring|changing|breaking))\b",
    r"\b(never (do|use|add|make|change|touch|modify) (this|that|it|them|these) again)\b",
    r"\b(please (just |for once |stop |don'?t |read |follow |listen ))\b",
    # Dismissal phrases
    r"\b(forget (it|this|that|everything|all of it))\b",
    r"\b(i (give|gave) up|giving up)\b",
    r"\b(waste of (my |our )?(time|effort))\b",
    r"\b(i'?m done (with this|with you|trying))\b",
]

_FRUSTRATION_RE = re.compile("|".join(_FRUSTRATION_PATTERNS), re.IGNORECASE)

# Layer 3: structural signals
_CAPS_RE = re.compile(r'\b[A-Z]{3,}\b')


def is_frustrated(text: str) -> bool:
    """Return True if the message contains any frustration signal."""
    lower = text.lower()

    # Layer 1: word/phrase list (fast substring check)
    for phrase in _FRUSTRATION_PHRASES:
        if phrase in lower:
            return True

    # Layer 2: structural regex
    if _FRUSTRATION_RE.search(text):
        return True

    # Layer 3: 2+ ALL CAPS words = yelling
    if len(_CAPS_RE.findall(text)) >= 2:
        return True

    # Layer 3: repeated punctuation (?? !! ?! !?)
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
