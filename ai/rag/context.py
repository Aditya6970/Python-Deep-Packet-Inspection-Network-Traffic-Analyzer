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
import math
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..schemas import Severity
from .chunking import KnowledgeChunk
from .documents import Category
from .retrieval import RetrievalReport, RetrievedChunk
from .signals import SignalType

__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MAX_ITEMS",
    "DEFAULT_MAX_TOKENS",
    "ENV_MAX_CHARS",
    "ENV_MAX_ITEMS",
    "ENV_MAX_TOKENS",
    "KNOWLEDGE_BLOCK_END",
    "KNOWLEDGE_BLOCK_START",
    "BudgetReason",
    "ExcludedKnowledge",
    "KnowledgeContext",
    "KnowledgeContextConfig",
    "KnowledgeItem",
    "build_knowledge_context",
    "estimate_tokens",
]

#: Delimiters around the reference block, matching the capture-data markers.
KNOWLEDGE_BLOCK_START: Final[str] = "===== BEGIN REFERENCE KNOWLEDGE ====="
KNOWLEDGE_BLOCK_END: Final[str] = "===== END REFERENCE KNOWLEDGE ====="

_REF_PATTERN: Final[re.Pattern[str]] = re.compile(r"^K[1-9][0-9]*$")

#: Any run of five or more '=' inside a chunk is broken up before rendering.
_FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"={5,}")


#: Characters per token used by :func:`estimate_tokens`.
#:
#: Deliberately pessimistic.  English prose runs nearer four characters per
#: token, but this corpus is full of things that tokenize far worse -- field
#: names like ``syn_ack_seen``, identifiers like ``T1071.004``, backticked code
#: fragments.  A budget that *underestimates* tokens is worthless, because the
#: request it approves is the one the provider rejects; overestimating only
#: costs a little unused headroom.
CHARS_PER_TOKEN: Final[float] = 3.5

#: Budget defaults, named so :meth:`KnowledgeContextConfig.from_env` can read
#: them -- a ``slots`` dataclass exposes descriptors rather than values on the
#: class, so ``cls.max_items`` is not the default.
DEFAULT_MAX_ITEMS: Final[int] = 4
DEFAULT_MAX_CHARS: Final[int] = 3000
DEFAULT_MAX_TOKENS: Final[int | None] = 900

ENV_MAX_ITEMS: Final[str] = "DPI_RAG_MAX_ITEMS"
ENV_MAX_CHARS: Final[str] = "DPI_RAG_MAX_CHARS"
ENV_MAX_TOKENS: Final[str] = "DPI_RAG_MAX_TOKENS"


def estimate_tokens(text: str) -> int:
    """Approximate the token count of ``text``, with no tokenizer dependency.

    A character ratio, not a tokenizer.  Loading a real tokenizer would mean
    pulling in ``tiktoken`` or the provider's own vocabulary -- a dependency,
    a download, and a number that is still only correct for one provider.  The
    ratio is provider-neutral, costs nothing, and is accurate enough for a
    budget whose job is to stay comfortably under a limit rather than to sit
    exactly on it.

    It is an **estimate** and is named one everywhere it is reported.
    """
    return _tokens_for(len(text))


def _tokens_for(characters: int) -> int:
    """Token estimate for a length, so a budget check needs no string."""
    return math.ceil(characters / CHARS_PER_TOKEN) if characters > 0 else 0


class BudgetReason(str, Enum):
    """Why an excerpt was left out of the prompt."""

    MAX_ITEMS = "max_items"
    MAX_CHARS = "max_chars"
    MAX_TOKENS = "max_total_tokens"


@dataclass(frozen=True, slots=True)
class KnowledgeContextConfig:
    """How much retrieved knowledge may reach a prompt.

    This is the second, prompt-facing limit: retrieval already caps how many
    chunks come back, and this decides how many of them a request can afford to
    carry.  It exists because a provider will reject an oversized request
    outright -- and when it does, the analysis is lost, not merely degraded.

    Provider-neutral on purpose.  The limits are expressed in excerpts,
    characters and estimated tokens, never in one vendor's quota; mapping a
    particular provider's ceiling onto these numbers is the operator's job,
    through configuration or the CLI.  Nothing in the RAG layer knows which
    provider it is feeding.

    The DPI capture data is **not** governed by this budget and never will be.
    Facts are the evidence; reference material is the commentary.  When
    something has to give, the commentary gives.
    """

    #: Most excerpts to include.
    max_items: int = DEFAULT_MAX_ITEMS

    #: Ceiling on the **rendered** size of the whole block, in characters.
    #:
    #: Rendered, not raw: the provenance header of each excerpt (document,
    #: section, category, similarity, citation, matched signals) is roughly
    #: 150 characters, so budgeting the chunk text alone understates what the
    #: request actually carries -- which is precisely the mistake that lets an
    #: oversized request through.
    max_chars: int = DEFAULT_MAX_CHARS

    #: Ceiling on the estimated tokens of the rendered block.
    #:
    #: ``None`` disables it and leaves ``max_chars`` in charge.  The two are
    #: near-equivalent by construction; the token form exists because provider
    #: limits are published in tokens, so an operator can transcribe a quota
    #: without converting it.
    max_total_tokens: int | None = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        if self.max_items < 0:
            raise ValueError("max_items must be zero or positive")
        if self.max_chars < 0:
            raise ValueError("max_chars must be zero or positive")
        if self.max_total_tokens is not None and self.max_total_tokens < 0:
            raise ValueError("max_total_tokens must be None or zero or positive")

    @classmethod
    def from_env(cls, **overrides: object) -> KnowledgeContextConfig:
        """Read the budget from the environment, explicit arguments winning.

        None of these is a secret -- they are three integers -- so unlike
        :class:`~ai.config.AIConfig` there is nothing here to mask.
        """
        values: dict[str, object] = {
            "max_items": _env_int(ENV_MAX_ITEMS, DEFAULT_MAX_ITEMS),
            "max_chars": _env_int(ENV_MAX_CHARS, DEFAULT_MAX_CHARS),
            "max_total_tokens": _env_int(ENV_MAX_TOKENS, DEFAULT_MAX_TOKENS),
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    def describe(self) -> str:
        """One line naming the active limits, for a report footer."""
        tokens = ("no token limit" if self.max_total_tokens is None
                  else f"{self.max_total_tokens} est. tokens")
        return f"{self.max_items} excerpts / {self.max_chars} chars / {tokens}"


def _env_int(name: str, default: int | None) -> int | None:
    """Read an integer environment variable, or return the default."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    text = raw.strip()
    if text.lower() in ("none", "off", "unlimited"):
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc


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


class ExcludedKnowledge(BaseModel):
    """One retrieved chunk the budget could not afford.

    Kept, not discarded.  A chunk that was retrieved and then dropped is a fact
    about this run: it says the budget bound, which excerpts the model never
    saw, and how close they were.  Silently losing that would make a thin
    answer indistinguishable from a well-supported one.

    Composition, as everywhere else -- the retrieved chunk is held whole rather
    than copied field by field, so an excluded excerpt reports exactly the same
    provenance an included one does.  It carries no ``[K#]`` label, and that is
    deliberate: labels are only ever assigned to excerpts the model was given,
    which is what keeps ``knowledge_refs`` checkable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieved: RetrievedChunk
    reason: BudgetReason

    @property
    def chunk_id(self) -> str:
        return self.retrieved.chunk_id

    @property
    def document_id(self) -> str:
        return self.retrieved.document_id

    @property
    def section(self) -> str:
        return self.retrieved.section

    @property
    def similarity(self) -> float:
        return self.retrieved.similarity

    @property
    def rank(self) -> int:
        """Its position in the retrieval ranking, which the label would have followed."""
        return self.retrieved.rank

    def citation(self) -> str:
        return self.retrieved.citation()

    def metadata(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "section": self.section,
            "citation": self.citation(),
            "similarity": round(self.similarity, 6),
            "retrieval_rank": self.rank,
            "excluded_by": self.reason.value,
        }


class KnowledgeContext(BaseModel):
    """The reference block, its items, and what was left out of it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[KnowledgeItem, ...] = ()
    text: str = Field(default="", description="The rendered block, or empty.")
    capped: bool = Field(
        default=False, description="True when some retrieved chunks were dropped."
    )
    dropped_items: int = Field(default=0, ge=0)
    excluded: tuple[ExcludedKnowledge, ...] = Field(
        default=(), description="Retrieved chunks the budget excluded, in rank order."
    )
    total_chars: int = Field(
        default=0, ge=0, description="Rendered characters of the block that was built."
    )
    estimated_tokens: int = Field(
        default=0, ge=0, description="Estimated tokens of the block. An estimate, not a count."
    )
    budget: str = Field(default="", description="The limits that were in force.")
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
        if self.dropped_items != len(self.excluded):
            raise ValueError("dropped_items must match the recorded exclusions")
        included = {item.chunk_id for item in self.items}
        if included & {dropped.chunk_id for dropped in self.excluded}:
            raise ValueError("a chunk cannot be both included and excluded")
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

    def excluded_chunk_ids(self) -> tuple[str, ...]:
        """Chunks retrieval found but the budget could not afford."""
        return tuple(dropped.chunk_id for dropped in self.excluded)

    def cited(self, knowledge_refs: Sequence[str]) -> tuple[KnowledgeItem, ...]:
        """The items the model actually cited, in label order."""
        wanted = set(knowledge_refs)
        return tuple(item for item in self.items if item.ref in wanted)

    def to_json(self, include_text: bool = False) -> str:
        """Stable JSON of the context metadata."""
        payload = {
            "item_count": len(self.items),
            "total_chars": self.total_chars,
            "estimated_tokens": self.estimated_tokens,
            "budget": self.budget,
            "capped": self.capped,
            "dropped_items": self.dropped_items,
            "notes": list(self.notes),
            "items": [
                dict(item.metadata(), **({"text": item.text} if include_text else {}))
                for item in self.items
            ],
            "excluded": [dropped.metadata() for dropped in self.excluded],
        }
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


def _render_block(items: Sequence[KnowledgeItem]) -> str:
    """Assemble the delimited block from already-selected items."""
    body = "\n\n".join(item.render() for item in items)
    return f"{KNOWLEDGE_BLOCK_START}\n{body}\n{KNOWLEDGE_BLOCK_END}"


#: Characters the delimiters and the blank lines between excerpts add.
_BLOCK_OVERHEAD: Final[int] = len(KNOWLEDGE_BLOCK_START) + len(KNOWLEDGE_BLOCK_END) + 2


def build_knowledge_context(
    retrieval: RetrievalReport | None,
    config: KnowledgeContextConfig | None = None,
) -> KnowledgeContext:
    """Select, within budget, the excerpts a prompt will carry.

    Runs **after** retrieval has ranked everything and **before** the prompt is
    built, so it changes what is affordable without touching what is relevant.

    The policy, in order:

    1. Walk the retrieved chunks in **retrieval rank order**. Nothing is
       re-sorted, re-scored or sampled -- the highest-ranked chunk is always
       considered first, and is always included if it fits.
    2. Measure each candidate as it will actually be **rendered**: provenance
       header plus text plus the separator. Budgeting the raw chunk text alone
       understates the request by roughly 150 characters per excerpt, which is
       exactly the error that lets an oversized request through.
    3. Include it when it fits inside every active limit; otherwise record it
       as :class:`ExcludedKnowledge` with the limit that stopped it, and
       **carry on to the next chunk** -- a later, shorter excerpt may still
       fit, and running out of room is not a reason to stop looking.
    4. Never truncate. A half-excerpt reads as complete and stops mid-argument,
       and its citation would point at text the reader cannot see. Excluding a
       lower-ranked excerpt is always preferable to corrupting a citation.

    The limits are hard. If even the first excerpt does not fit, none is
    supplied and the run proceeds with no knowledge -- which the report states
    plainly. That is the point: a budget that is exceeded when it matters is
    not a budget, and a rejected request loses the analysis entirely rather
    than degrading it.

    Deterministic: same retrieval report and same config, same block, byte for
    byte, including which excerpts were excluded and why.
    """
    cfg = config or KnowledgeContextConfig()
    chunks = list(retrieval.chunks) if retrieval is not None else []

    items: list[KnowledgeItem] = []
    excluded: list[ExcludedKnowledge] = []
    used_chars = _BLOCK_OVERHEAD if chunks else 0

    for chunk in chunks:
        candidate = KnowledgeItem(ref=f"K{len(items) + 1}",
                                  retrieved=_renumbered(chunk, len(items)))
        # Size as rendered, including the blank line that will separate it
        # from the previous excerpt.
        size = len(candidate.render()) + (2 if items else 0)

        reason = _rejection(cfg, len(items), used_chars, size)
        if reason is not None:
            excluded.append(ExcludedKnowledge(retrieved=chunk, reason=reason))
            continue

        items.append(candidate)
        used_chars += size

    if not items:
        text = ""
        estimated = 0
    else:
        text = _render_block(items)
        used_chars = len(text)
        estimated = estimate_tokens(text)

    notes = _budget_notes(cfg, items, excluded)

    return KnowledgeContext(
        items=tuple(items),
        text=text,
        capped=bool(excluded),
        dropped_items=len(excluded),
        excluded=tuple(excluded),
        total_chars=used_chars if items else 0,
        estimated_tokens=estimated,
        budget=cfg.describe(),
        notes=tuple(notes),
    )


def _rejection(
    cfg: KnowledgeContextConfig, included: int, used_chars: int, size: int
) -> BudgetReason | None:
    """Which limit, if any, this candidate would break.

    Checked in declaration order so the reported reason is stable: a chunk that
    breaks both the item count and the character budget is always reported
    against the item count.
    """
    if included >= cfg.max_items:
        return BudgetReason.MAX_ITEMS
    if used_chars + size > cfg.max_chars:
        return BudgetReason.MAX_CHARS
    if (cfg.max_total_tokens is not None
            and _tokens_for(used_chars + size) > cfg.max_total_tokens):
        return BudgetReason.MAX_TOKENS
    return None


def _budget_notes(
    cfg: KnowledgeContextConfig,
    items: Sequence[KnowledgeItem],
    excluded: Sequence[ExcludedKnowledge],
) -> list[str]:
    """Human-readable disclosure of what the budget did."""
    if not excluded:
        return []

    counts: dict[str, int] = {}
    for dropped in excluded:
        counts[dropped.reason.value] = counts.get(dropped.reason.value, 0) + 1
    breakdown = ", ".join(f"{count} by {name}" for name, count in sorted(counts.items()))

    notes = [
        f"{len(excluded)} retrieved excerpt(s) were excluded by the context budget "
        f"({cfg.describe()}): {breakdown}. No excerpt was truncated."
    ]
    if not items:
        notes.append(
            "Nothing fitted the budget, so no reference knowledge was supplied. "
            "Raise DPI_RAG_MAX_CHARS or DPI_RAG_MAX_ITEMS to include some."
        )
    return notes


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
