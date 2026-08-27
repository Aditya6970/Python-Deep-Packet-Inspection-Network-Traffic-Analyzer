"""Turning retrieved chunks into the reference block a prompt can carry.

One serializer, one format, one place
-------------------------------------
:func:`build_knowledge_context` is the only thing in the project that renders
retrieved knowledge as text.  Everything downstream -- the prompt, the console
report, ``--show-knowledge`` -- reads the same :class:`KnowledgeContext`, so
the numbering a reviewer sees on screen is the numbering the model was given.
If those two ever disagreed, every citation in the output would point at the
wrong document.

Labels
------
Items are numbered ``K1``, ``K2``, ... **in retrieval order** -- the ranking
:mod:`ai.rag.retrieval` already fixed, not dictionary order and not a re-sort.
The label is positional, so the same retrieval always yields the same labels,
and :meth:`~ai.schemas.AnalysisResult.validate_knowledge_references` can check
a citation against the exact set that was supplied.

Nothing here talks to a model, a store or a network.  It is text assembly over
objects that already exist.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..schemas import Severity
from .chunking import KnowledgeChunk
from .documents import Category
from .retrieval import RetrievalReport, RetrievedChunk
from .signals import SignalType

__all__ = [
    "KNOWLEDGE_BLOCK_END",
    "KNOWLEDGE_BLOCK_START",
    "KnowledgeContext",
    "KnowledgeContextConfig",
    "KnowledgeItem",
    "build_knowledge_context",
]

#: Delimiters around the reference block, matching the capture-data markers.
KNOWLEDGE_BLOCK_START: Final[str] = "===== BEGIN REFERENCE KNOWLEDGE ====="
KNOWLEDGE_BLOCK_END: Final[str] = "===== END REFERENCE KNOWLEDGE ====="

_REF_PATTERN: Final[re.Pattern[str]] = re.compile(r"^K[1-9][0-9]*$")

#: Any run of five or more '=' inside a chunk is broken up before rendering.
_FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"={5,}")


@dataclass(frozen=True, slots=True)
class KnowledgeContextConfig:
    """How much retrieved knowledge may reach a prompt.

    Deliberately conservative.  Retrieval already caps how many chunks come
    back; this is the second, prompt-facing limit, and it exists because the
    number of chunks that is useful to a reader is smaller than the number a
    context window can physically hold.
    """

    #: Most excerpts to include.
    max_items: int = 6

    #: Ceiling on the total characters of chunk text (excluding headers).
    #:
    #: Roughly 1500 tokens of English at four characters per token -- enough
    #: for several sections, small enough to leave the capture JSON dominant in
    #: the prompt, which is the intended balance: the capture is the evidence.
    max_chars: int = 6000

    def __post_init__(self) -> None:
        if self.max_items < 0:
            raise ValueError("max_items must be zero or positive")
        if self.max_chars < 0:
            raise ValueError("max_chars must be zero or positive")


class KnowledgeItem(BaseModel):
    """One numbered excerpt, as the model will see it.

    Composition again: the retrieved chunk is held whole rather than having its
    document, section, title, category, text and similarity copied out beside
    the label.  The only genuinely new fact here is the label itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: str = Field(description='Positional label: "K1", "K2", ...')
    retrieved: RetrievedChunk

    @field_validator("ref")
    @classmethod
    def _ref_shape(cls, v: str) -> str:
        if not _REF_PATTERN.match(v):
            raise ValueError(f"ref {v!r} is malformed; expected K1, K2, ...")
        return v

    @model_validator(mode="after")
    def _label_matches_rank(self) -> KnowledgeItem:
        """The label is the rank plus one -- not an independent counter.

        Tying them together means a reordering can never silently renumber the
        block while leaving the ranks alone.
        """
        if int(self.ref[1:]) != self.retrieved.rank + 1:
            raise ValueError(
                f"ref {self.ref} does not match retrieval rank {self.retrieved.rank}"
            )
        return self

    # -- provenance, read through --------------------------------------------
    @property
    def chunk(self) -> KnowledgeChunk:
        return self.retrieved.chunk

    @property
    def chunk_id(self) -> str:
        return self.retrieved.chunk_id

    @property
    def document_id(self) -> str:
        return self.retrieved.document_id

    @property
    def title(self) -> str:
        return self.retrieved.title

    @property
    def category(self) -> Category:
        return self.retrieved.category

    @property
    def section(self) -> str:
        return self.retrieved.section

    @property
    def text(self) -> str:
        return self.retrieved.text

    @property
    def similarity(self) -> float:
        return self.retrieved.similarity

    @property
    def severity_hint(self) -> Severity:
        return self.chunk.severity_hint

    @property
    def matched_signal_types(self) -> tuple[SignalType, ...]:
        return self.retrieved.matched_signal_types

    def citation(self) -> str:
        return self.retrieved.citation()

    def render(self) -> str:
        """The excerpt as it appears in the prompt.

        Provenance first so a reader can judge the source before the text, and
        so a citation in the output can be traced back by eye.  ``Text:`` comes
        last because it is the only multi-line field.
        """
        signals = ", ".join(item.value for item in self.matched_signal_types) or "none"
        lines = [
            f"[{self.ref}]",
            f"Document: {self.document_id}",
            f"Section: {self.section}",
            f"Category: {self.category.value}",
            f"Similarity: {self.similarity:.4f}",
            f"Citation: {self.citation()}",
            f"Matched signals: {signals}",
            "Text:",
            _neutralise_fences(self.text),
        ]
        return "\n".join(lines)

    def metadata(self) -> dict[str, object]:
        """Flat view for reports and JSON export.  Never includes the text."""
        return {
            "ref": self.ref,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "category": self.category.value,
            "section": self.section,
            "similarity": round(self.similarity, 6),
            "citation": self.citation(),
            "matched_signal_types": [i.value for i in self.matched_signal_types],
            "licence": self.chunk.licence,
        }


def _neutralise_fences(text: str) -> str:
    """Break up any run of ``=`` long enough to imitate a block delimiter.

    The corpus is curated and reviewed, so this should never fire.  It is here
    because the delimiters are the only thing separating reference text from
    instructions, and a defence that depends on nobody ever writing a row of
    equals signs in a Markdown table is not a defence.
    """
    return _FENCE_PATTERN.sub(lambda m: " ".join("=" * len(m.group())), text)


class KnowledgeContext(BaseModel):
    """The reference block, its items, and what was left out of it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[KnowledgeItem, ...] = ()
    text: str = Field(default="", description="The rendered block, or empty.")
    capped: bool = Field(
        default=False, description="True when some retrieved chunks were dropped."
    )
    dropped_items: int = Field(default=0, ge=0)
    total_chars: int = Field(default=0, ge=0, description="Characters of chunk text used.")
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> KnowledgeContext:
        refs = [item.ref for item in self.items]
        if refs != [f"K{i + 1}" for i in range(len(self.items))]:
            raise ValueError("items must be labelled K1..Kn in order")
        if len(set(item.chunk_id for item in self.items)) != len(self.items):
            raise ValueError("the same chunk appears twice in the context")
        if self.capped != (self.dropped_items > 0):
            raise ValueError("capped must be true exactly when items were dropped")
        if self.items and not self.text:
            raise ValueError("a context with items must render to text")
        if self.text and not self.items:
            raise ValueError("a context with no items must render to nothing")
        return self

    def __bool__(self) -> bool:
        """Truthy only when there is knowledge to supply."""
        return bool(self.items)

    def refs(self) -> tuple[str, ...]:
        """The labels supplied to the model -- what a citation is checked against."""
        return tuple(item.ref for item in self.items)

    def by_ref(self, ref: str) -> KnowledgeItem | None:
        return next((item for item in self.items if item.ref == ref), None)

    def cited(self, knowledge_refs: Sequence[str]) -> tuple[KnowledgeItem, ...]:
        """The items the model actually cited, in label order."""
        wanted = set(knowledge_refs)
        return tuple(item for item in self.items if item.ref in wanted)

    def to_json(self, include_text: bool = False) -> str:
        """Stable JSON of the context metadata."""
        payload = {
            "item_count": len(self.items),
            "total_chars": self.total_chars,
            "capped": self.capped,
            "dropped_items": self.dropped_items,
            "notes": list(self.notes),
            "items": [
                dict(item.metadata(), **({"text": item.text} if include_text else {}))
                for item in self.items
            ],
        }
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


#: An empty context, for the many paths where no knowledge is available.
_EMPTY = KnowledgeContext()


def build_knowledge_context(
    retrieval: RetrievalReport | None,
    config: KnowledgeContextConfig | None = None,
) -> KnowledgeContext:
    """Render a retrieval result as the numbered reference block.

    Items are taken **in retrieval order** and labelled by position.  Chunks
    are included whole or not at all: when the character budget runs out the
    remaining, lower-ranked chunks are dropped rather than truncated, because
    half a section of security guidance is worse than none -- it reads as
    complete and stops mid-argument.

    Deterministic: the same retrieval report always produces the same block,
    byte for byte.
    """
    cfg = config or KnowledgeContextConfig()

    if retrieval is None or not retrieval.chunks:
        return _EMPTY

    items: list[KnowledgeItem] = []
    used_chars = 0
    dropped = 0
    dropped_for_chars = 0

    for chunk in retrieval.chunks:
        if len(items) >= cfg.max_items:
            dropped += 1
            continue
        if used_chars + len(chunk.text) > cfg.max_chars and items:
            # Keep going rather than break: a later, shorter chunk may still
            # fit, and dropping by size is not a reason to drop by rank too.
            dropped += 1
            dropped_for_chars += 1
            continue
        items.append(KnowledgeItem(ref=f"K{len(items) + 1}",
                                   retrieved=_renumbered(chunk, len(items))))
        used_chars += len(chunk.text)

    if not items:
        return _EMPTY

    notes: list[str] = []
    if dropped:
        notes.append(
            f"{dropped} retrieved chunk(s) were not included: the prompt limit is "
            f"{cfg.max_items} excerpts and {cfg.max_chars} characters."
        )
    if dropped_for_chars:
        notes.append(
            f"{dropped_for_chars} chunk(s) were dropped whole to stay within the "
            "character budget; no excerpt was truncated."
        )

    body = "\n\n".join(item.render() for item in items)
    text = f"{KNOWLEDGE_BLOCK_START}\n{body}\n{KNOWLEDGE_BLOCK_END}"

    return KnowledgeContext(
        items=tuple(items),
        text=text,
        capped=dropped > 0,
        dropped_items=dropped,
        total_chars=used_chars,
        notes=tuple(notes),
    )


def _renumbered(chunk: RetrievedChunk, position: int) -> RetrievedChunk:
    """Return ``chunk`` with its rank set to its position in the context.

    Dropping a chunk for the character budget leaves a gap in the ranks, and a
    label must always equal ``rank + 1`` -- otherwise K3 could carry rank 4 and
    a reviewer comparing the two would be reading a different excerpt from the
    model.  Ranks are renumbered here rather than the invariant being relaxed.
    """
    if chunk.rank == position:
        return chunk
    return chunk.model_copy(update={"rank": position})
