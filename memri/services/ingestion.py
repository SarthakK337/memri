"""Ingestion pipeline: conversation → episode → facts → entities → graph."""

from typing import List, Dict, Optional
from datetime import datetime
import re

from ..graph.types import MemoryNode, Edge, NodeType, EdgeType
from ..graph.store import GraphStore
from ..graph.embeddings import EmbeddingStore
from ..llm.graph_adapter import GraphLLMAdapter as LLMProvider
from ..storage.layer2 import Layer2Store
from .query_processor import HAS_SPACY

# kept for backward compat; new code uses SESSION_SUMMARY_STATIC + generate_cached
SESSION_SUMMARY_PROMPT = """Summarize this conversation session in one factual sentence (max 40 words).
Focus on who did what, where, and when. Be specific about names, places, and events.

{conversation}

One sentence summary:"""

# Only these spaCy NER types become graph entity nodes.
# Everything else (lunch, concert, gift, music) stays as text in fact content.
# BM25 finds those by keyword — they don't need graph nodes.
ALLOWED_ENTITY_TYPES = {"PERSON", "GPE", "LOC", "ORG", "WORK_OF_ART", "FAC"}


def _extract_proper_entities(fact_content: str) -> List[str]:
    """Use spaCy NER to extract only proper named entities from fact text."""
    if not HAS_SPACY:
        return []
    try:
        from .query_processor import _get_nlp
        doc = _get_nlp()(fact_content)
        entities = []
        for ent in doc.ents:
            name = ent.text.strip().strip(",'\"")
            # Filter out garbage: too short, contains spaces with verbs, all lowercase, etc.
            if not name or len(name) < 2:
                continue
            if not name[0].isupper():
                continue
            # Skip if it looks like a phrase (verb present) e.g. "Oliver hid"
            ent_doc = doc.vocab.strings  # cheap check via token count
            words = name.split()
            if len(words) > 3:
                continue
            if ent.label_ in ALLOWED_ENTITY_TYPES:
                entities.append(name)
        return list(set(entities))
    except Exception:
        return []

# ── Static (cacheable) portions of each ingestion prompt ──────────────────────
# These never change across sessions and are sent to Gemini context cache once.

FACT_EXTRACTION_STATIC = """You are a precise fact extractor. Extract EVERY distinct piece of information from a conversation as atomic facts.

RULES:
- Each fact must be self-contained and specific
- Preserve ALL names, dates, numbers, locations, food items, preferences
- Extract MORE facts rather than fewer (aim for 20-30 per session)
- Include seemingly minor details — they may be asked about later
- ALWAYS capture exact titles of books, songs, movies, shows, artworks in quotes
- ALWAYS capture exact names of pets, people, places, brands
- NEVER paraphrase proper nouns: if someone says "Sweden" write "Sweden" not "her home country". If they say "Austin" write "Austin" not "that city".
- For relative dates ("yesterday", "last Saturday", "last week"): include BOTH the relative phrase AND the absolute date calculated from the DATE header
  Example: if DATE header is "25 May 2023" and text says "last Saturday" → write "last Saturday (20 May 2023)"
- For FUTURE references ("next month", "this month", "next year"): compute the actual month/year from the DATE header
  Example: if DATE header is "8 May 2023" and text says "next month" → write "next month (June 2023)"
- State exact counts using DIGITS: "Melanie has 3 children", "went to beach 2 times in 2023"
- Always emit a RELATIONSHIP STATUS fact when mentioned: "Caroline is single", "Melanie is married to David"

COMPOUND EVENT RULE — most important rule:
When one utterance contains multiple activities or sub-events joined by commas or "and", emit ONE fact per activity. Each fact must say WHAT was done AS PART OF the parent event.

BAD (one compound fact):
  "Melanie and family went camping, roasted marshmallows, told stories, and hiked"

GOOD (four atomic facts):
  "Melanie and family went camping last week (the week before 27 June 2023)"
  "Melanie and family roasted marshmallows during camping last week (the week before 27 June 2023)"
  "Melanie and family told stories around the campfire during camping last week"
  "Melanie and family went hiking during camping last week"

COUNT RULE:
- ALWAYS use digits, not words: "2 times" not "twice", "7 years" not "seven years"
- When a count is mentioned, emit both a narrative fact AND a standalone count fact
- "went to beach once or twice a year" → also emit "Melanie goes to beach 1-2 times per year"
- "7 years" without a start year → also emit "started approximately [session_year - 7]"

COLLECTION RULE:
When someone lists owned items, emit one fact per item AND one summary count fact:
  "she has 2 cats named Luna and Bailey, and a dog named Oliver"
  → "Melanie has a cat named Luna"
  → "Melanie has a cat named Bailey"
  → "Melanie has a dog named Oliver"
  → "Melanie has 2 cats and 1 dog"

RELATIONSHIP STATE:
On any relationship or status mention, emit: "<A> is <relation> of <B>" or "<A> works at <org>".
These are state facts — emit them even if they seem obvious.

SPECIFIC DETAILS — always extract as standalone facts:
- Favorite food/drink/snack: "<person>'s favorite food is <exact item>"
- Stress relievers / coping mechanisms: "<person> relieves stress by <exact activity>"
- Recurring frustrations / habits: "<person> repeatedly/always <exact behavior>"
- Medical conditions / ailments: "<person> was diagnosed with / has <exact condition name>"
- Each item in a list of suggestions, recommendations, or examples gets its own fact

EMOTION: For each fact, identify the emotional tone of the speaker when stating it.
- emotion_label: one of neutral | sad | angry | frustrated | anxious | grieving | proud | excited | joyful
- emotion_intensity: 0.0 (none) to 1.0 (overwhelming)

For each fact provide:
- importance (0.0-1.0): life events, named items, specific details = high; casual filler = low
- emotional_weight: same as emotion_intensity
- emotion_label: see above
- emotion_intensity: 0.0-1.0
- entities: people, places, named things
- temporal_reference: the relative phrase exactly as spoken ("last week", "yesterday") — NOT the resolved date

Respond ONLY as a JSON array:
[
  {
    "content": "exact detailed fact here",
    "importance": 0.7,
    "emotional_weight": 0.3,
    "emotion_label": "sad",
    "emotion_intensity": 0.3,
    "entities": ["Sarah", "CorePower"],
    "temporal_reference": "last week"
  }
]"""

DETAIL_EXTRACTION_STATIC = """You are a detail extractor. You already have general facts. Now extract SPECIFIC DETAILS a quiz might ask about.

Focus ONLY on what is NOT already captured:
- EXACT titles of books, songs, movies, TV shows, albums, artworks (use quotes)
- EXACT names of restaurants, places, stores, brands, organizations
- Pet names and types ("guinea pig named Oscar", "dog named Bailey")
- Gift descriptions: exact item + who gave it + occasion
- Counts and quantities ("3 children", "2 cats and a dog", "went twice in 2023")
- Exact quotes attributed to a specific person
- Any named entity (person, place, thing) visible anywhere in the conversation

Respond as JSON array ([] if nothing new):
[
  {
    "content": "Melanie's favorite childhood book is \\"Charlotte's Web\\"",
    "importance": 0.6,
    "emotional_weight": 0.1,
    "entities": ["Melanie", "Charlotte's Web"],
    "temporal_reference": null
  }
]"""

CAUSAL_LINKING_STATIC = """You identify causal relationships between facts from the same conversation.

Given a numbered list of facts, find cause→effect pairs.
Respond as JSON array of pairs (empty array if none):
[
  {"cause": 1, "effect": 2, "relationship": "caused"},
  {"cause": 2, "effect": 3, "relationship": "coping_with"}
]

Only "caused" and "coping_with" are valid relationships. Empty array is fine."""

SESSION_SUMMARY_STATIC = """Summarize a conversation session in one factual sentence (max 40 words).
Focus on who did what, where, and when. Be specific about names, places, and events.
Output one sentence only."""

CAUSAL_EDGE_MAP = {
    "caused": EdgeType.CAUSED,
    "coping_with": EdgeType.COPING_WITH,
}

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _parse_session_date(session_date_str: str):
    """Parse session_date (may include time like '1:56 pm on 8 May, 2023') → (date_str, year, month_idx, day).

    Returns (clean_date_str, year, month_idx_1based, day) or (session_date_str, None, None, None).
    """
    if not session_date_str:
        return session_date_str, None, None, None
    month_pat = "|".join(_MONTH_NAMES)
    m = re.search(
        rf'(\d{{1,2}})\s+({month_pat}),?\s+(\d{{4}})',
        session_date_str, re.IGNORECASE,
    )
    if m:
        day = int(m.group(1))
        month_name = m.group(2).capitalize()
        year = int(m.group(3))
        month_idx = _MONTH_NAMES.index(month_name) + 1
        clean = f"{day} {month_name} {year}"
        return clean, year, month_idx, day
    return session_date_str, None, None, None


def _add_months(year: int, month: int, delta: int) -> tuple:
    month += delta
    while month > 12:
        month -= 12
        year += 1
    while month < 1:
        month += 12
        year -= 1
    return year, month


def resolve_temporal_reference(ref: str, session_date_str: str) -> Optional[str]:
    """Combine relative phrase + session_date → resolved human-readable form.

    Returns None if ref is already absolute or no pattern matches.
    """
    if not ref or not session_date_str:
        return None

    clean_date, year, month_idx, day = _parse_session_date(session_date_str)
    ref_lower = ref.lower().strip()

    # Compute-based resolutions (return actual dates/years, not relative phrases)
    if year is not None:
        if re.search(r"\blast\s+year\b", ref_lower):
            return str(year - 1)
        if re.search(r"\bthis\s+year\b", ref_lower):
            return str(year)
        if re.search(r"\bnext\s+year\b", ref_lower):
            return str(year + 1)
        if re.search(r"\blast\s+month\b", ref_lower):
            py, pm = _add_months(year, month_idx, -1)
            return f"{_MONTH_NAMES[pm-1]} {py}"
        if re.search(r"\bthis\s+month\b", ref_lower):
            return f"{_MONTH_NAMES[month_idx-1]} {year}"
        if re.search(r"\bnext\s+month\b", ref_lower):
            ny, nm = _add_months(year, month_idx, 1)
            return f"{_MONTH_NAMES[nm-1]} {ny}"
        if re.search(r"\bthis\s+summer\b", ref_lower):
            return f"Summer {year}"
        if re.search(r"\bthis\s+fall\b|\bthis\s+autumn\b", ref_lower):
            return f"Fall {year}"
        if re.search(r"\bthis\s+spring\b", ref_lower):
            return f"Spring {year}"
        if re.search(r"\bthis\s+winter\b", ref_lower):
            return f"Winter {year}"

    # Relative-phrase resolutions (keep relative form, use clean date as anchor)
    PATTERNS = [
        (r"\byesterday\b",              f"the day before {clean_date}"),
        (r"\blast\s+week\b",            f"the week before {clean_date}"),
        (r"\blast\s+weekend\b",         f"the weekend before {clean_date}"),
        (r"\blast\s+friday\b",          f"the Friday before {clean_date}"),
        (r"\blast\s+saturday\b",        f"the Saturday before {clean_date}"),
        (r"\blast\s+sunday\b",          f"the Sunday before {clean_date}"),
        (r"\blast\s+monday\b",          f"the Monday before {clean_date}"),
        (r"\blast\s+tuesday\b",         f"the Tuesday before {clean_date}"),
        (r"\blast\s+wednesday\b",       f"the Wednesday before {clean_date}"),
        (r"\blast\s+thursday\b",        f"the Thursday before {clean_date}"),
        (r"\btwo\s+weekends?\s+ago\b",  f"two weekends before {clean_date}"),
        (r"\ba\s+few\s+weeks?\s+ago\b", f"a few weeks before {clean_date}"),
        (r"\brecently\b",               f"around {clean_date}"),
        (r"\ba\s+few\s+days?\s+ago\b",  f"a few days before {clean_date}"),
        (r"\bearlier\s+this\s+week\b",  f"earlier the week of {clean_date}"),
        (r"\bthis\s+morning\b",         f"the morning of {clean_date}"),
        (r"\bthis\s+weekend\b",         f"the weekend of {clean_date}"),
    ]
    for pattern, resolved in PATTERNS:
        if re.search(pattern, ref_lower):
            return resolved
    return None


_RELATIVE_KEYWORDS = re.compile(
    r"\b(yesterday|last\s+week|last\s+weekend|last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|two\s+weekends?\s+ago|a\s+few\s+weeks?\s+ago|recently|last\s+month|last\s+year|next\s+month|this\s+month"
    r"|this\s+year|next\s+year|this\s+summer|this\s+fall|this\s+autumn|this\s+spring|this\s+winter"
    r"|a\s+few\s+days?\s+ago|earlier\s+this\s+week|this\s+morning|this\s+weekend)\b",
    re.IGNORECASE,
)


class IngestionService:
    def __init__(self, graph: GraphStore, embeddings: EmbeddingStore, llm: LLMProvider, layer2: Layer2Store = None):
        self.graph = graph
        self.embeddings = embeddings
        self.llm = llm
        self.layer2 = layer2

    async def ingest_conversation(self, conversation: str, session_index: int = 0, session_date: str = None) -> Dict:
        """Full pipeline: raw conversation → structured memory graph."""

        # Step 1: Store episode node
        episode = MemoryNode(node_type=NodeType.EPISODE, content=conversation, importance=0.5)
        self.graph.add_node(episode)

        # Store raw text in Layer 2
        if self.layer2:
            self.layer2.store_episode(episode.id, conversation, session_index, session_date)

        # Step 2: Extract facts via LLM (static rules cached, only conversation sent each time)
        try:
            extracted = await self.llm.generate_json_cached(
                static=FACT_EXTRACTION_STATIC,
                dynamic=f"Conversation:\n{conversation}",
            )
            if not isinstance(extracted, list):
                extracted = []
        except Exception as e:
            print(f"[ingestion] Fact extraction failed: {e}")
            extracted = []

        # Step 2b: Second pass for specific details (books, songs, quotes, gifts, etc.)
        try:
            existing_facts_text = "\n".join(f"- {item.get('content', '')}" for item in extracted)
            details = await self.llm.generate_json_cached(
                static=DETAIL_EXTRACTION_STATIC,
                dynamic=(
                    f"Conversation:\n{conversation}\n\n"
                    f"Already extracted facts (do NOT repeat):\n{existing_facts_text or '(none yet)'}"
                ),
            )
            if isinstance(details, list):
                extracted.extend(details)
        except Exception as e:
            print(f"[ingestion] Detail extraction failed: {e}")

        NEGATIVE_EMOTIONS = {"sad", "angry", "frustrated", "anxious", "grieving"}

        # Step 3: Create fact nodes + entity linking
        fact_nodes: List[MemoryNode] = []
        for item in extracted:
            if not isinstance(item, dict):
                continue

            emotion_label = str(item.get("emotion_label", "neutral") or "neutral").lower()
            try:
                emotion_intensity = float(item.get("emotion_intensity", 0.0) or 0.0)
            except (TypeError, ValueError):
                emotion_intensity = 0.0
            try:
                base_importance = float(item.get("importance", 0.5) or 0.5)
            except (TypeError, ValueError):
                base_importance = 0.5

            # Boost importance for negative-emotion facts
            if emotion_label in NEGATIVE_EMOTIONS and emotion_intensity > 0.1:
                base_importance = min(1.0, base_importance + 0.25)

            # Coerce content to string — LLM occasionally returns a list or None
            raw_content = item.get("content", "")
            if isinstance(raw_content, list):
                raw_content = " ".join(str(x) for x in raw_content)
            content = str(raw_content or "").strip()
            if not content:
                continue

            # Coerce temporal_reference to string
            raw_ref = item.get("temporal_reference", "")
            temporal_ref = str(raw_ref).strip() if raw_ref is not None else ""

            # Resolve relative temporal references → append to content so BM25 indexes them
            if session_date and temporal_ref and _RELATIVE_KEYWORDS.search(temporal_ref):
                resolved = resolve_temporal_reference(temporal_ref, session_date)
                if resolved and resolved not in content:
                    content = content.rstrip(".") + f" ({resolved})"
            # Also scan content itself for relative keywords not captured in temporal_reference
            elif session_date and _RELATIVE_KEYWORDS.search(content):
                match = _RELATIVE_KEYWORDS.search(content)
                if match:
                    resolved = resolve_temporal_reference(match.group(0), session_date)
                    if resolved and resolved not in content:
                        content = content.rstrip(".") + f" ({resolved})"

            fact = MemoryNode(
                node_type=NodeType.FACT,
                content=content,
                importance=base_importance,
                emotional_weight=emotion_intensity,
                source_episode_id=episode.id,
                temporal_reference=item.get("temporal_reference"),
                session_index=session_index,
                session_date=session_date,
            )
            # Store emotion metadata in content_abstract for future use
            fact.content_abstract = {
                "emotion_label": emotion_label,
                "emotion_intensity": emotion_intensity,
            }
            self.graph.add_node(fact)
            fact_nodes.append(fact)

            # Link fact → episode
            self.graph.add_edge(Edge(
                source_id=fact.id,
                target_id=episode.id,
                edge_type=EdgeType.DERIVED_FROM,
            ))

            # Link fact → entities (spaCy NER: PERSON/GPE/LOC/ORG/WORK_OF_ART/FAC only)
            proper_entities = _extract_proper_entities(fact.content)
            # Fallback: use LLM entity list filtered to multi-word or capitalized names
            # when spaCy is not available
            if not proper_entities and not HAS_SPACY:
                for name in item.get("entities", []) or []:
                    name = str(name).strip()
                    if name and (name[0].isupper() or len(name.split()) > 1):
                        proper_entities.append(name)
            for entity_name in proper_entities:
                entity_node = self.graph.find_or_create_entity(entity_name)
                self.graph.add_edge(Edge(
                    source_id=fact.id,
                    target_id=entity_node.id,
                    edge_type=EdgeType.BELONGS_TO,
                ))

        # Step 4: Temporal linking — chain new facts with HAPPENED_AFTER
        for i in range(1, len(fact_nodes)):
            self.graph.add_edge(Edge(
                source_id=fact_nodes[i].id,
                target_id=fact_nodes[i - 1].id,
                edge_type=EdgeType.HAPPENED_AFTER,
            ))

        # Also link first new fact to most recent prior fact (cross-episode temporal chain)
        prior_facts = [
            n for n in self.graph.get_nodes_by_type(NodeType.FACT)
            if n.id not in {f.id for f in fact_nodes}
        ]
        if prior_facts and fact_nodes:
            prior_facts.sort(key=lambda n: n.created_at, reverse=True)
            self.graph.add_edge(Edge(
                source_id=fact_nodes[0].id,
                target_id=prior_facts[0].id,
                edge_type=EdgeType.HAPPENED_AFTER,
            ))

        # Step 5: Causal linking via LLM
        if len(fact_nodes) >= 2:
            facts_list = "\n".join(
                f"{i + 1}. {f.content}" for i, f in enumerate(fact_nodes)
            )
            try:
                causal = await self.llm.generate_json_cached(
                    static=CAUSAL_LINKING_STATIC,
                    dynamic=f"Facts:\n{facts_list}",
                )
                if isinstance(causal, list):
                    for rel in causal:
                        cause_idx = int(rel.get("cause", 0)) - 1
                        effect_idx = int(rel.get("effect", 0)) - 1
                        rel_type = rel.get("relationship", "")
                        edge_type = CAUSAL_EDGE_MAP.get(rel_type)
                        if (
                            edge_type
                            and 0 <= cause_idx < len(fact_nodes)
                            and 0 <= effect_idx < len(fact_nodes)
                        ):
                            self.graph.add_edge(Edge(
                                source_id=fact_nodes[cause_idx].id,
                                target_id=fact_nodes[effect_idx].id,
                                edge_type=edge_type,
                            ))
            except Exception as e:
                print(f"[ingestion] Causal linking failed: {e}")

        # Step 6: Generate embeddings for facts and entities
        new_entity_ids = set()
        for fact in fact_nodes:
            self.embeddings.add(fact.id, fact.content, {"node_type": "fact"})
            fact.embedding_id = fact.id

            # Embed any newly created entities connected to this fact
            for neighbor_id in self.graph.get_neighbors(fact.id, edge_type=EdgeType.BELONGS_TO):
                if neighbor_id not in new_entity_ids:
                    entity = self.graph.get_node(neighbor_id)
                    if entity and entity.node_type == NodeType.ENTITY:
                        self.embeddings.add(entity.id, entity.content, {"node_type": "entity"})
                        entity.embedding_id = entity.id
                        new_entity_ids.add(neighbor_id)

        # Step 6b: Detect superseding — mark old facts replaced by new ones
        if session_index and fact_nodes:
            self._detect_superseded(fact_nodes, session_index)

        # Step 7: Generate and store session summary in Layer 2
        if self.layer2:
            try:
                summary_text = await self.llm.generate_cached(
                    static=SESSION_SUMMARY_STATIC,
                    dynamic=conversation[:3000],
                )
                summary_text = summary_text.strip()
                # Embed the summary using a dedicated collection ID
                summary_embed_id = f"summary_{episode.id}"
                self.embeddings.add(summary_embed_id, summary_text, {"node_type": "session_summary"})
                self.layer2.store_summary(episode.id, summary_text, summary_embed_id)
            except Exception as e:
                print(f"[ingestion] Session summary generation failed: {e}")

        return {
            "episode_id": episode.id,
            "facts_extracted": len(fact_nodes),
            "entities_linked": len(new_entity_ids),
        }

    def _detect_superseded(self, new_facts: List[MemoryNode], session_index: int) -> None:
        """Mark existing facts as superseded when a new fact replaces the same state.

        Uses embedding similarity + entity overlap to avoid false positives.
        Only marks facts from strictly earlier sessions.
        """
        new_ids = {f.id for f in new_facts}
        existing = {
            n.id: n for n in self.graph.get_nodes_by_type(NodeType.FACT)
            if n.id not in new_ids
            and not n.is_superseded
            and (n.session_index is None or n.session_index < session_index)
        }
        if not existing:
            return

        for new_fact in new_facts:
            similar = self.embeddings.search(new_fact.content, n_results=6)
            new_entity_ids = set(self.graph.get_neighbors(new_fact.id, edge_type=EdgeType.BELONGS_TO))

            for old_id, score in similar:
                if score < 0.88:
                    continue
                old_fact = existing.get(old_id)
                if not old_fact:
                    continue

                old_entity_ids = set(self.graph.get_neighbors(old_fact.id, edge_type=EdgeType.BELONGS_TO))

                if new_entity_ids and old_entity_ids:
                    if not (new_entity_ids & old_entity_ids):
                        continue  # different subjects
                elif not new_entity_ids and not old_entity_ids:
                    if score < 0.92:  # tighter without entity confirmation
                        continue

                old_fact.is_superseded = True
                existing.pop(old_id)  # don't supersede same fact twice
                self.graph.add_edge(Edge(
                    source_id=new_fact.id,
                    target_id=old_fact.id,
                    edge_type=EdgeType.SUPERSEDES,
                ))
