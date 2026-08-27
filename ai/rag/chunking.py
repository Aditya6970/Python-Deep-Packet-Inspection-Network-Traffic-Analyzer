"""Deterministic, provenance-preserving chunking of the knowledge corpus.

What this does
--------------
Turns the validated :class:`~ai.rag.documents.KnowledgeDocument` objects
produced by step 1 into :class:`KnowledgeChunk` objects: the units that will
later be embedded, indexed, retrieved and cited.  Nothing here embeds,
indexes, retrieves, ranks or calls anything.  There is no network access, no
tokenizer, and no dependency beyond the standard library and pydantic.

The chunker consumes **validated documents, never raw Markdown**.  Parsing,
front-matter validation and the six-section template check all belong to
:mod:`ai.rag.documents`; by the time a document reaches this module its
sections are already guaranteed to exist, to be non-empty and to be in order.
That is why a malformed document can never produce a chunk: it never gets
past the loader.

Strategy: sections first, structure second
-----------------------------------------
Every knowledge document is authored to a fixed six-section template, and a
section *is* a complete thought — "Indicators for DNS tunneling" is
self-contained and citable on its own.  So the primary unit is the section,
and most sections become exactly one chunk.

Only when a section exceeds :attr:`ChunkConfig.max_chars` is it split, and the
split walks a hierarchy of natural boundaries, preferring the largest
structure that fits:

1. **Paragraph breaks** — a blank line.
2. **List-item starts** — so a long bulleted section splits between items
   rather than through one.
3. **Sentence ends** — ``.``, ``!`` or ``?`` followed by whitespace.
4. **A whitespace-aligned hard cut** — the documented last resort, reached
   only by a single "sentence" longer than the whole budget.

Structure is preferred, but not at any price.  A boundary is only taken if it
fills at least :attr:`ChunkConfig.min_fill_ratio` of the budget; otherwise the
next weaker kind of boundary is considered.  Without that rule a section that
opens with a one-line introduction followed by a long list would cut at the
blank line after the introduction — technically the strongest boundary
available, and a 66-character chunk that carries no content.  Measured on the
real corpus, that is exactly what happened before the rule was added.

Splitting is done by computing *offsets* into the section text and slicing,
so a chunk body is always a verbatim substring of its source section.  That
matters for provenance: a citation can be checked against the document by
eye, character for character.

Overlap
-------
Overlap exists to stop a split from orphaning the sentence that gives the next
chunk its subject.  It is therefore applied **only where a split actually
happened**: a section that fits in one chunk gets none, which is the common
case in this corpus.  Where it applies, the tail of the previous chunk is
prepended, aligned to a sentence boundary, and the amount is recorded in
:attr:`KnowledgeChunk.overlap_chars` so a consumer knows exactly how many
leading characters are duplicated context rather than new material.

Determinism
-----------
Identical input produces byte-identical output: same chunks, same ids, same
order, every run, on every platform.  Nothing consults the clock, the
filesystem's iteration order, a random seed, or a hash whose value varies per
process (Python's ``hash()`` is salted; SHA-256 is not, which is why ids use
it).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Final, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..schemas import Severity
from .documents import (
    REQUIRED_SECTIONS,
    Category,
    KnowledgeDocument,
    load_corpus,
)

__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_OVERLAP_CHARS",
    "OVERLAP_SEPARATOR",
    "Boundary",
    "ChunkConfig",
    "KnowledgeChunk",
    "chunk_corpus",
    "chunk_document",
    "chunk_statistics",
    "section_slug",
    "serialize_chunks",
]


# ===========================================================================
# Configuration
# ===========================================================================
#: Maximum characters in one chunk, overlap included.
#:
#: Chosen for the embedding model planned for step 3,
#: ``BAAI/bge-small-en-v1.5``, which truncates at 512 tokens.  English prose
#: runs roughly 4 characters per token, so 512 tokens is about 2000
#: characters — but that ratio degrades on exactly the text this corpus
#: contains: field names like ``syn_ack_seen``, identifiers like ``T1071.004``
#: and backticked code fragments tokenize far worse than plain prose.  1400
#: characters keeps a comfortable margin under the ceiling, leaves room for
#: the heading path a retriever will prepend, and stays large enough that the
#: authored sections of this corpus survive intact.
#:
#: Deliberately measured against the real corpus: at 1400, thirty-five of
#: thirty-six sections are a single chunk and exactly one splits, so the split
#: path is exercised by real content rather than only by test fixtures.
#:
#: Characters, not tokens, on purpose — a token count would mean a tokenizer
#: dependency, and step 2 stays dependency-free.
DEFAULT_MAX_CHARS: Final[int] = 1400

#: Characters of the previous chunk repeated at the head of the next one,
#: applied only when a section is actually split.  Roughly one to two
#: sentences: enough to carry the subject across a boundary, small enough that
#: it does not meaningfully dilute the chunk's own content.
DEFAULT_OVERLAP_CHARS: Final[int] = 200

#: Smallest share of the budget a structural boundary may leave in a chunk.
#:
#: Guards against a strong-but-useless split point: a section that opens with
#: a one-line introduction before a long list offers a paragraph break after
#: about sixty characters, and taking it would emit a chunk with no content in
#: it.  At 0.6 the chunker skips that boundary and looks for a weaker one that
#: actually fills the chunk.
DEFAULT_MIN_FILL_RATIO: Final[float] = 0.6

#: Separator inserted between the overlap tail and the new body.
OVERLAP_SEPARATOR: Final[str] = "\n\n"


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Chunking parameters.  Immutable, so a run cannot drift mid-corpus."""

    max_chars: int = DEFAULT_MAX_CHARS
    overlap_chars: int = DEFAULT_OVERLAP_CHARS
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO

    @property
    def min_fill_chars(self) -> int:
        """Smallest chunk a structural boundary is allowed to produce."""
        return int(self.max_chars * self.min_fill_ratio)

    def __post_init__(self) -> None:
        if self.max_chars < 200:
            raise ValueError("max_chars must be at least 200 to hold a usable chunk")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative")
        if not 0.0 <= self.min_fill_ratio < 1.0:
            raise ValueError("min_fill_ratio must be in [0.0, 1.0)")
        if self.overlap_chars * 2 >= self.max_chars:
            raise ValueError(
                "overlap_chars must be well under half of max_chars, or a split "
                "section would be mostly repetition"
            )


# ===========================================================================
# Boundary detection
# ===========================================================================
class Boundary:
    """Rank of a candidate split point.  Lower is more preferred."""

    PARAGRAPH: Final[int] = 1
    LIST_ITEM: Final[int] = 2
    SENTENCE: Final[int] = 3
    HARD: Final[int] = 4

    NAMES: Final[dict[int, str]] = {
        PARAGRAPH: "paragraph",
        LIST_ITEM: "list_item",
        SENTENCE: "sentence",
        HARD: "hard",
    }


_PARAGRAPH_BREAK: Final[re.Pattern[str]] = re.compile(r"\n[ \t]*\n")
_LIST_ITEM_START: Final[re.Pattern[str]] = re.compile(r"(?m)^[ \t]*(?:[-*+]|\d+\.)\s")
_SENTENCE_END: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])[ \t]*\n|(?<=[.!?])[ \t]+")
_WHITESPACE_RUN: Final[re.Pattern[str]] = re.compile(r"\s+")
_SLUG_STRIP: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
_CHUNK_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9-]+#[a-z0-9-]+#\d{2}-[0-9a-f]{12}$"
)


def section_slug(section: str) -> str:
    """Slugify a section heading for use inside a chunk id.

    ``"What the DPI engine can observe"`` -> ``"what-the-dpi-engine-can-observe"``.
    The six headings are fixed, so this mapping is stable for the life of the
    corpus.
    """
    return _SLUG_STRIP.sub("-", section.lower()).strip("-")


def _boundaries(text: str) -> dict[int, int]:
    """Map candidate offsets in ``text`` to their :class:`Boundary` rank.

    An offset is the index at which a new chunk would *start*.  Where two
    kinds of boundary coincide, the stronger (lower) rank wins.
    """
    found: dict[int, int] = {}

    def record(offset: int, rank: int) -> None:
        if 0 < offset < len(text):
            found[offset] = min(rank, found.get(offset, rank))

    for match in _PARAGRAPH_BREAK.finditer(text):
        record(match.end(), Boundary.PARAGRAPH)
    for match in _LIST_ITEM_START.finditer(text):
        record(match.start(), Boundary.LIST_ITEM)
    for match in _SENTENCE_END.finditer(text):
        record(match.end(), Boundary.SENTENCE)

    return found


def _best_cut(
    boundaries: dict[int, int], start: int, limit: int, min_fill: int
) -> tuple[int, int] | None:
    """Pick the split point in ``(start, limit]`` that preserves most structure.

    Strongest rank first, but only among boundaries that leave at least
    ``min_fill`` characters in the chunk; within a rank, the largest offset
    wins so chunks stay as full as the structure allows.  If no rank offers a
    well-filled boundary the constraint is dropped and the largest boundary of
    the strongest available rank is taken — a short chunk beats a mid-sentence
    cut.
    """
    in_window = [
        (offset, rank) for offset, rank in boundaries.items() if start < offset <= limit
    ]
    if not in_window:
        return None

    for rank in (Boundary.PARAGRAPH, Boundary.LIST_ITEM, Boundary.SENTENCE):
        filled = [
            offset
            for offset, candidate_rank in in_window
            if candidate_rank == rank and offset - start >= min_fill
        ]
        if filled:
            return max(filled), rank

    best_rank = min(rank for _, rank in in_window)
    return max(offset for offset, rank in in_window if rank == best_rank), best_rank


def _hard_cut(text: str, start: int, limit: int) -> int:
    """Last-resort cut, pulled back to a whitespace run where one exists.

    Reached only when a single unbroken stretch of text longer than the whole
    budget contains no paragraph, list or sentence boundary — a run-on line, a
    long URL, a wide table row.  Splitting mid-word is avoided when possible;
    splitting mid-sentence here is unavoidable by construction.
    """
    window = text[start:limit]
    match = None
    for match in _WHITESPACE_RUN.finditer(window):
        pass
    if match is not None and match.end() > 0:
        return start + match.end()
    return limit


def _overlap_tail(previous: str, budget: int) -> str:
    """Return the tail of ``previous`` to repeat at the head of the next chunk.

    Aligned to a sentence boundary when one falls inside the budget, so the
    overlap is a readable fragment rather than a severed clause.  Capped at
    half the previous chunk as well as at ``budget``, so a short chunk is
    never almost entirely duplicated.
    """
    if budget <= 0 or not previous:
        return ""

    allowance = min(budget, len(previous) // 2)
    if allowance <= 0:
        return ""

    floor = len(previous) - allowance
    candidates = [m.end() for m in _SENTENCE_END.finditer(previous) if m.end() >= floor]
    if candidates:
        return previous[min(candidates):].strip()

    window = previous[floor:]
    match = _WHITESPACE_RUN.search(window)
    if match is not None:
        return window[match.end():].strip()
    return window.strip()


# ===========================================================================
# Chunk model
# ===========================================================================
class KnowledgeChunk(BaseModel):
    """One retrievable, citable unit of the knowledge corpus.

    Every field is either the chunk's own text, a deterministic property of
    that text, or metadata copied verbatim from the source document.  Nothing
    is inferred, scored or invented here — this model is a carrier, and its
    job is to make a later retrieval hit answer "which document and which
    section did this come from?" without a second lookup.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # -- identity -----------------------------------------------------------
    chunk_id: str = Field(
        description="Deterministic: <document_id>#<section-slug>#<index>-<content hash>."
    )
    content_sha256: str = Field(
        min_length=64, max_length=64, description="SHA-256 of the normalised chunk text."
    )

    # -- provenance ---------------------------------------------------------
    document_id: str = Field(min_length=3, description="id of the source document.")
    document_sha256: str = Field(
        min_length=64,
        max_length=64,
        description="Hash of the whole source file, linking a chunk to an exact revision.",
    )
    document_version: str
    document_updated: date
    relative_path: str = Field(description="Path within knowledge/, POSIX-style.")
    title: str
    category: Category
    section: str = Field(description="One of the six required section headings.")
    section_index: int = Field(ge=0, le=len(REQUIRED_SECTIONS) - 1)
    heading_path: str = Field(description='"<title> > <section>", for display and citation.')

    # -- position -----------------------------------------------------------
    chunk_index: int = Field(ge=0, description="Index within this section.")
    chunk_count: int = Field(ge=1, description="Total chunks produced from this section.")

    # -- content ------------------------------------------------------------
    text: str = Field(min_length=1)
    char_count: int = Field(ge=1)
    overlap_chars: int = Field(
        ge=0, description="Leading characters repeated from the previous chunk."
    )

    # -- inherited document metadata ---------------------------------------
    keywords: list[str]
    applies_to: list[str]
    mitre: list[str] = Field(default_factory=list)
    severity_hint: Severity
    sources: list[str] = Field(min_length=1)
    licence: str = Field(min_length=2)

    # -- validators ---------------------------------------------------------
    @field_validator("text")
    @classmethod
    def _text_has_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunk text is empty or whitespace only")
        return v

    @field_validator("section")
    @classmethod
    def _known_section(cls, v: str) -> str:
        if v not in REQUIRED_SECTIONS:
            raise ValueError(
                f"section {v!r} is not one of the six required sections "
                f"{list(REQUIRED_SECTIONS)}"
            )
        return v

    @field_validator("chunk_id")
    @classmethod
    def _well_formed_id(cls, v: str) -> str:
        if not _CHUNK_ID_PATTERN.match(v):
            raise ValueError(
                f"chunk_id {v!r} is malformed; expected "
                "<document-id>#<section-slug>#<NN>-<12 hex digits>"
            )
        return v

    @model_validator(mode="after")
    def _internally_consistent(self) -> KnowledgeChunk:
        """Catch a chunk whose derived fields disagree with its own text."""
        if self.section_index != REQUIRED_SECTIONS.index(self.section):
            raise ValueError("section_index does not match section")
        if self.char_count != len(self.text):
            raise ValueError("char_count does not match len(text)")
        if self.chunk_index >= self.chunk_count:
            raise ValueError("chunk_index is out of range for chunk_count")
        if self.overlap_chars > self.char_count:
            raise ValueError("overlap_chars exceeds the chunk length")
        if self.chunk_index == 0 and self.overlap_chars:
            raise ValueError("the first chunk of a section cannot have overlap")
        if not self.chunk_id.startswith(f"{self.document_id}#"):
            raise ValueError("chunk_id does not begin with its document_id")
        return self

    # -- helpers ------------------------------------------------------------
    def body(self) -> str:
        """The chunk's own text, with any repeated overlap removed."""
        if not self.overlap_chars:
            return self.text
        return self.text[self.overlap_chars + len(OVERLAP_SEPARATOR):]

    def citation(self) -> str:
        """A short human-readable provenance string."""
        suffix = f" [{self.chunk_index + 1}/{self.chunk_count}]" if self.chunk_count > 1 else ""
        return f"{self.document_id} / {self.section}{suffix}"


# ===========================================================================
# Chunking
# ===========================================================================
def _normalise(text: str) -> str:
    """Collapse whitespace, for hashing only.

    Ids are computed from normalised text so that a reflow or a trailing-space
    edit does not churn every id downstream, while any real change to the
    words still produces a different id.
    """
    return _WHITESPACE_RUN.sub(" ", text).strip()


def _content_hash(document_id: str, section: str, index: int, text: str) -> str:
    """Hash the identity of one chunk.

    SHA-256 rather than :func:`hash`, whose value is salted per process and
    would make ids differ between runs.
    """
    payload = "\n".join([document_id, section, str(index), _normalise(text)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_section(text: str, config: ChunkConfig) -> list[tuple[str, int]]:
    """Split one section into ``(chunk_text, overlap_chars)`` pairs.

    A section that fits returns a single pair with zero overlap.  Otherwise
    the section is cut at the strongest available boundary that fits the
    budget, and each subsequent chunk is prefixed with the tail of its
    predecessor.
    """
    body = text.strip()
    if len(body) <= config.max_chars:
        return [(body, 0)]

    boundaries = _boundaries(body)
    pieces: list[tuple[str, int]] = []
    start = 0

    while start < len(body):
        if pieces:
            tail = _overlap_tail(pieces[-1][0], config.overlap_chars)
            prefix = tail + OVERLAP_SEPARATOR if tail else ""
        else:
            prefix = ""

        budget = config.max_chars - len(prefix)
        remaining = len(body) - start

        if remaining <= budget:
            piece = body[start:]
            start = len(body)
        else:
            limit = start + budget
            cut = _best_cut(boundaries, start, limit, config.min_fill_chars)
            end = cut[0] if cut is not None else _hard_cut(body, start, limit)
            if end <= start:  # pragma: no cover - guarded against, never expected
                end = limit
            piece = body[start:end]
            start = end

        piece = piece.strip()
        if not piece:
            continue

        if prefix:
            pieces.append((prefix + piece, len(prefix) - len(OVERLAP_SEPARATOR)))
        else:
            pieces.append((piece, 0))

    return pieces


def chunk_document(
    document: KnowledgeDocument, config: ChunkConfig | None = None
) -> tuple[KnowledgeChunk, ...]:
    """Chunk one validated document, in section order.

    Sections are visited in :data:`~ai.rag.documents.REQUIRED_SECTIONS` order
    rather than in dict order, so the result does not depend on how the
    document happened to be parsed.
    """
    cfg = config or ChunkConfig()
    meta = document.metadata
    chunks: list[KnowledgeChunk] = []

    for section_index, section in enumerate(REQUIRED_SECTIONS):
        pieces = _split_section(document.section(section), cfg)
        total = len(pieces)
        slug = section_slug(section)

        for index, (text, overlap) in enumerate(pieces):
            digest = _content_hash(meta.id, section, index, text)
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{meta.id}#{slug}#{index:02d}-{digest[:12]}",
                    content_sha256=digest,
                    document_id=meta.id,
                    document_sha256=document.sha256,
                    document_version=meta.version,
                    document_updated=meta.updated,
                    relative_path=document.relative_path,
                    title=meta.title,
                    category=meta.category,
                    section=section,
                    section_index=section_index,
                    heading_path=f"{meta.title} > {section}",
                    chunk_index=index,
                    chunk_count=total,
                    text=text,
                    char_count=len(text),
                    overlap_chars=overlap,
                    keywords=list(meta.keywords),
                    applies_to=list(meta.applies_to),
                    mitre=list(meta.mitre),
                    severity_hint=meta.severity_hint,
                    sources=list(meta.sources),
                    licence=meta.licence,
                )
            )

    return tuple(chunks)


def chunk_corpus(
    documents: Sequence[KnowledgeDocument] | None = None,
    config: ChunkConfig | None = None,
) -> tuple[KnowledgeChunk, ...]:
    """Chunk a whole corpus, preserving corpus order.

    Ordering is ``corpus order -> section order -> chunk index``.  The corpus
    order comes from :func:`~ai.rag.documents.load_corpus`, which sorts by
    category then id, so nothing here depends on filesystem iteration.

    Passing ``documents`` explicitly is how the tests chunk a temporary corpus
    without touching the real one.
    """
    corpus = load_corpus() if documents is None else tuple(documents)
    chunks: list[KnowledgeChunk] = []
    for document in corpus:
        chunks.extend(chunk_document(document, config))
    return tuple(chunks)


# ===========================================================================
# Reporting
# ===========================================================================
def chunk_statistics(chunks: Iterable[KnowledgeChunk]) -> dict[str, object]:
    """Summarise a chunk set.  Pure arithmetic; no side effects."""
    items = list(chunks)
    if not items:
        return {
            "documents": 0,
            "chunks": 0,
            "max_chars": 0,
            "min_chars": 0,
            "mean_chars": 0.0,
            "total_chars": 0,
            "split_sections": 0,
            "chunks_with_overlap": 0,
            "by_category": {},
        }

    sizes = [c.char_count for c in items]
    return {
        "documents": len({c.document_id for c in items}),
        "chunks": len(items),
        "max_chars": max(sizes),
        "min_chars": min(sizes),
        "mean_chars": round(sum(sizes) / len(sizes), 1),
        "total_chars": sum(sizes),
        "split_sections": len(
            {(c.document_id, c.section) for c in items if c.chunk_count > 1}
        ),
        "chunks_with_overlap": sum(1 for c in items if c.overlap_chars),
        "by_category": {
            category.value: sum(1 for c in items if c.category is category)
            for category in Category
        },
    }


def serialize_chunks(chunks: Iterable[KnowledgeChunk]) -> str:
    """Serialise chunks to stable JSON.

    Sorted keys and a fixed separator, so two runs over unchanged input
    produce byte-identical text — the simplest possible proof of determinism,
    and the format a later index build would persist.
    """
    payload = [chunk.model_dump(mode="json") for chunk in chunks]
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


# ===========================================================================
# Manual check:  python -m ai.rag.chunking
# ===========================================================================
if __name__ == "__main__":  # pragma: no cover - manual check
    import sys

    from .documents import KnowledgeError

    settings = ChunkConfig()
    try:
        produced = chunk_corpus(config=settings)
    except KnowledgeError as error:
        print(f"corpus failed to load\n  {error}")
        sys.exit(1)

    stats = chunk_statistics(produced)
    print(f"documents:        {stats['documents']}")
    print(f"chunks:           {stats['chunks']}")
    print(f"max chunk size:   {stats['max_chars']} chars (limit {settings.max_chars})")
    print(f"min chunk size:   {stats['min_chars']} chars")
    print(f"average size:     {stats['mean_chars']} chars")
    print(f"sections split:   {stats['split_sections']}")
    print(f"chunks w/overlap: {stats['chunks_with_overlap']} "
          f"(overlap budget {settings.overlap_chars} chars)")
    print("by category:      " + ", ".join(
        f"{name}={count}" for name, count in stats["by_category"].items()  # type: ignore[union-attr]
    ))

    split = [c for c in produced if c.chunk_count > 1]
    if split:
        print("\nsplit sections:")
        for chunk in split:
            print(f"  {chunk.citation():<52} {chunk.char_count:>5} chars"
                  f"  overlap={chunk.overlap_chars}")
