"""QueryIntent classifier — heuristic-only, no LLM call."""

from dataclasses import dataclass, field
from typing import List, Optional
import re


@dataclass
class QueryIntent:
    answer_shape: str           # "list" | "date" | "duration" | "count" | "yesno" | "short_phrase"
    entities: List[str] = field(default_factory=list)
    temporal_cue: Optional[str] = None
    predicate: Optional[str] = None
    is_recency: bool = False
    is_aggregate: bool = False


def classify_query(question: str) -> QueryIntent:
    """Heuristic-first classifier. Pure regex + keyword matching, no LLM."""
    q = question.lower().strip()

    # --- answer_shape ---
    if re.search(r"\bwhen\b|\bhow long\b|\bsince when\b|\bwhat (year|month|day|date)\b", q):
        answer_shape = "date"
    elif re.search(r"\bhow (many|often|much|many times|frequently)\b|\bhow many\b", q):
        answer_shape = "count"
    elif re.search(
        # "what [adj*] <list-noun>" — allow up to 2 adjectives between what and noun
        r"\bwhat\s+(?:\w+\s+){0,2}(activities|things|events|items|places|foods|movies|books|songs|shows"
        r"|hobbies|kinds?|types?|subjects?|suggestions?|examples?|incidents?|relievers?|habits?|issues?)\b"
        r"|\blist\b|\bname all\b|\bwhat all\b"
        # "what did/does/has X [verb]" — X is a list of items
        r"|\bwhat\b.{0,40}\b(consider|suggest|recommend|experience|face|tell|give|say|include|involve)\b"
        # "who did X tell / who did X meet" — multiple people answer
        r"|\bwho\s+(did|does|has)\s+\w+\s+(tell|meet|invite|inform|contact|know)\b"
        # "what is a stress reliever / coping habit / hobby" — singular category queries
        r"|\bwhat\s+is\s+a?\s*\w*\s*(stress\s+reliev|coping|frustrat|habit|hobby|activit)\w*\b",
        q
    ):
        answer_shape = "list"
    elif re.search(r"\b(would|is|was|does|did|has|have|could|can|will)\s+\w+\s+(likely|probably|prefer|enjoy|like)\b|\bwould .+ (do|enjoy|like|prefer|go)\b", q):
        answer_shape = "yesno"
    elif re.search(r"\bwhere\b", q):
        answer_shape = "short_phrase"
    elif re.search(r"\bwhy\b|\bhow did\b|\bhow does\b", q):
        answer_shape = "short_phrase"
    else:
        answer_shape = "short_phrase"

    # --- temporal cues ---
    temporal_cue = None
    is_recency = False

    if re.search(r"\brecently\b|\blately\b|\bin the past\b|\bmost recent\b|\blatest\b|\blast time\b", q):
        is_recency = True
        temporal_cue = "recently"
    elif m := re.search(r"\bin (\d{4})\b", q):
        temporal_cue = m.group(0)
    elif m := re.search(r"\b(last year|last month|last week|last summer|last winter|last spring|last fall)\b", q):
        temporal_cue = m.group(0)
    elif m := re.search(r"\bin (january|february|march|april|may|june|july|august|september|october|november|december)\b", q):
        temporal_cue = m.group(0)

    # --- aggregate detection ---
    is_aggregate = bool(re.search(
        r"\ball\b|\bevery\b|\bover the years\b|\bthroughout\b|\beach\b|\bany\b|\bever\b|\btotal\b"
        r"|\bkinds? of\b|\btypes? of\b|\brecurring\b|\bpersistent\b|\bvarious\b|\bdifferent\b"
        r"|\blist\b|\bname all\b|\bwhat all\b"
        r"|\bwhat\b.{0,40}\b(consider|suggest|recommend|experience|face|tell|give|say|include|involve)\b"
        r"|\bwhat\s+(?:\w+\s+){0,2}(activities|hobbies|things|events|items|foods|incidents?|relievers?|habits?|issues?)\b"
        r"|\bwhat\s+is\s+a?\s*\w*\s*(stress\s+reliev|coping|frustrat|habit|hobby|activit)\w*\b",
        q
    ))

    # --- predicate ---
    predicate = None
    for kw in ("feel", "felt", "emotion", "own", "has", "have", "go", "went", "visit", "work", "live", "say", "said", "do", "did"):
        if kw in q.split():
            predicate = kw
            break

    return QueryIntent(
        answer_shape=answer_shape,
        temporal_cue=temporal_cue,
        is_recency=is_recency,
        is_aggregate=is_aggregate,
        predicate=predicate,
    )
