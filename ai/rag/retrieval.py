"""Deterministic, signal-driven retrieval of knowledge chunks.

What this does
--------------
Turns a :class:`~ai.rag.signals.SignalReport` into a ranked list of
:class:`~ai.rag.chunking.KnowledgeChunk` objects drawn from the vector store::

    SignalReport -> queries -> query vectors -> VectorStore.search -> ranked chunks

and stops there.  It builds no prompt, contacts no provider and asks no model
to write anything.  Step 6 ends at *"here are the most relevant knowledge
chunks for the observed signals"*; assembling them into a prompt is step 7.

Retrieval is driven by signals, not by prose
--------------------------------------------
The obvious design -- concatenate the whole report into one blob, embed it,
search once -- retrieves badly.  A capture that fired ``dns_high_volume`` and
``plaintext_http`` produces a blended vector that sits between both topics and
is strongly similar to neither.  So each signal gets **its own query**, each
query searches independently, and the results are merged afterwards.  One
extra capture-level query carries the profile, so protocol and port context is
retrievable even when no individual signal is about it.

Nothing here re-implements similarity.  :class:`~ai.rag.vector_store.VectorStore`
owns cosine and its tie-breaking rule; this module calls ``search`` once per
query and merges.

Queries can never carry capture-derived text
--------------------------------------------
Hostnames reach this project from TLS SNI, which is supplied by whoever opened
the connection -- attacker-controlled input.  A query built by pasting them in
would be a prompt-injection channel that opens the moment step 7 puts
retrieved text near a model.

The rule that prevents it is mechanical rather than careful: **only numeric and
boolean evidence values are rendered into a query.**  String and list evidence
-- ``top_parent_domain``, ``distinct_ports``, ``grouped_by`` -- is never read,
and no query template interpolates a hostname, an address or a capture name.
Every assembled query is then checked against an address and domain pattern
before it is used, so a leak introduced by a future edit raises rather than
ships.

Determinism
-----------
The same :class:`~ai.rag.signals.SignalReport` always produces byte-identical
query text, and the same store always returns the same ranking.  No clock, no
randomness, no iteration over unordered structures.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .chunking import KnowledgeChunk
from .documents import Category
from .embeddings import EmbeddingModel
from .signals import Signal, SignalReport, SignalType
from .vector_store import VectorStore

__all__ = [
    "BGE_QUERY_PREFIX",
    "CAPTURE_QUERY_LABEL",
    "QUERY_TEMPLATES",
    "RETRIEVAL_SCHEMA_VERSION",
    "ModelMismatchError",
    "RetrievalConfig",
    "RetrievalError",
    "RetrievalReport",
    "RetrievedChunk",
    "SignalQuery",
    "apply_query_prefix",
    "build_queries",
    "retrieve_for_signals",
]

#: Bumped when the retrieval report shape changes.
RETRIEVAL_SCHEMA_VERSION: Final[str] = "1.0"

#: The instruction prefix ``bge`` models expect on the **query** side.
#:
#: ``bge`` is trained asymmetrically: passages are embedded as they are, while
#: queries carry this sentence.  Step 3 embeds passages and therefore applies
#: nothing; the prefix belongs here, on the query side, and is applied exactly
#: once by :func:`apply_query_prefix`.
#:
#: Omitting it costs recall without raising anything, which is precisely the
#: kind of bug that survives to production, so :class:`SignalQuery` validates
#: that the embedded text is the prefix followed by the query and nothing else.
BGE_QUERY_PREFIX: Final[str] = "Represent this sentence for searching relevant passages: "

#: Label used for the one capture-wide query.
CAPTURE_QUERY_LABEL: Final[str] = "capture"

_LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(capture|signal:[a-z][a-z0-9_]*)$")
_IPV4_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_DOMAIN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b[a-z0-9][a-z0-9-]*\.[a-z]{2,}\b")


# ===========================================================================
# Errors
# ===========================================================================
class RetrievalError(Exception):
    """Base class for retrieval problems."""


class ModelMismatchError(RetrievalError):
    """The query embedding does not belong in this index.

    Raised when the embedder's model differs from the model the store was
    built with.  Two models produce vectors in two unrelated spaces, and the
    cosine scores between them are arithmetic without meaning -- so this is
    refused rather than ranked.
    """


# ===========================================================================
# Query templates
# ===========================================================================
#: One fixed phrase per signal type, describing the *topic* to retrieve.
#:
#: Templates rather than the signal's own summary alone, because a summary is
#: mostly numbers ("9 of 19 flows (47%)") and numbers embed poorly.  The
#: template supplies the vocabulary a matching knowledge document would use;
#: the summary is appended after it so the observation stays attached.
#:
#: Every template is written by hand here.  None interpolates anything from a
#: capture.
QUERY_TEMPLATES: Final[dict[SignalType, str]] = {
    SignalType.DNS_HIGH_VOLUME: (
        "high volume of DNS queries over UDP port 53, normal DNS resolution "
        "behaviour, and when DNS query volume is meaningful"
    ),
    SignalType.DNS_HIGH_CARDINALITY: (
        "many distinct DNS query names under one parent domain, name cardinality "
        "and caching, DNS tunneling indicators, and benign services that generate "
        "unique hostnames"
    ),
    SignalType.DNS_ANOMALOUS_LABEL: (
        "long high-entropy DNS subdomain labels, encoded data in query names, "
        "label length limits, and machine-generated hostnames that look encoded"
    ),
    SignalType.SCAN_PORT_FANOUT: (
        "one source contacting many destination ports, network service scanning "
        "and reconnaissance, and authorised vulnerability scanners"
    ),
    SignalType.SCAN_HALF_OPEN: (
        "TCP SYN sent without a SYN-ACK reply, half-open connections, closed "
        "ports and firewall drops"
    ),
    SignalType.UNKNOWN_APP_SHARE: (
        "traffic the deep packet inspection classifier could not identify, "
        "triaging unknown application traffic, and why encrypted flows lack a "
        "server name"
    ),
    SignalType.TLS_WITHOUT_SNI: (
        "TLS connections with no Server Name Indication, encrypted client hello, "
        "and what the engine can still observe without a hostname"
    ),
    SignalType.PLAINTEXT_HTTP: (
        "unencrypted HTTP on port 80, cleartext protocols, and what a plaintext "
        "request exposes"
    ),
    SignalType.QUIC_PRESENT: (
        "QUIC over UDP port 443, HTTP/3 transport, and how QUIC appears to deep "
        "packet inspection"
    ),
    SignalType.UPLOAD_ASYMMETRY: (
        "flows that send far more than they receive, upload-dominant traffic, "
        "data exfiltration indicators, and benign upload-heavy applications"
    ),
    SignalType.NONSTANDARD_PORT_EGRESS: (
        "outbound traffic to uncommon destination ports, non-standard port usage, "
        "and applications that use high dynamic ports"
    ),
    SignalType.BLOCKED_TRAFFIC_PRESENT: (
        "traffic blocked by configured rules, engine verdicts and blocking policy, "
        "and what a dropped flow does and does not mean"
    ),
    SignalType.BASELINE_WEB_BROWSING: (
        "ordinary web browsing baseline, content delivery networks and multi-host "
        "page loads, and why normal browsing generates many short flows"
    ),
}

# A template for every signal type, and no template for anything else.
assert set(QUERY_TEMPLATES) == set(SignalType), (
    "QUERY_TEMPLATES and SignalType have diverged: "
    f"{sorted({t.value for t in SignalType} ^ {t.value for t in QUERY_TEMPLATES})}"
)


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Retrieval parameters.  Immutable, so one run cannot drift mid-report."""

    #: Chunks each individual query may contribute before merging.
    #: ``0`` means none, matching :meth:`VectorStore.search`'s ``top_k=0``.
    per_query_top_k: int = 4

    #: Chunks in the final merged result.  ``0`` means none, for the same
    #: reason; there is no "unlimited" sentinel -- pass a number larger than
    #: the index instead, so a zero never has two meanings in one config.
    final_top_k: int = 8

    #: Optional cosine floor applied by the vector store, per query.
    #:
    #: ``None`` by default, deliberately.  A sensible floor depends on how the
    #: real model scores this corpus, and measuring that distribution is the
    #: evaluation step's job.  The parameter exists and is tested; the default
    #: does not guess a number it cannot justify.
    min_similarity: float | None = None

    #: Most chunks one document may contribute to the final result; ``None``
    #: disables the cap.  Without it a single long document can occupy every
    #: slot, since its sections are all similar to the same query.
    max_per_document: int | None = 2

    #: Include the capture-wide profile query alongside the per-signal ones.
    include_capture_query: bool = True

    def __post_init__(self) -> None:
        if self.per_query_top_k < 0:
            raise ValueError("per_query_top_k must be zero or positive")
        if self.final_top_k < 0:
            raise ValueError("final_top_k must be zero or positive")
        if self.max_per_document is not None and self.max_per_document < 1:
            raise ValueError("max_per_document must be None (no cap) or at least 1")
        if self.min_similarity is not None and not -1.0 <= self.min_similarity <= 1.0:
            raise ValueError(
                f"min_similarity must be within [-1, 1], got {self.min_similarity}"
            )

    def as_dict(self) -> dict[str, float | int | bool | str]:
        """Parameters used, for the report -- so a result can be reproduced."""
        return {
            "per_query_top_k": self.per_query_top_k,
            "final_top_k": self.final_top_k,
            "min_similarity": "none" if self.min_similarity is None
            else float(self.min_similarity),
            "max_per_document": "none" if self.max_per_document is None
            else self.max_per_document,
            "include_capture_query": self.include_capture_query,
        }


# ===========================================================================
# Query construction
# ===========================================================================
def apply_query_prefix(text: str) -> str:
    """Prepend the bge query instruction, exactly once.

    Raises rather than silently skipping if the text already carries it: a
    doubled prefix and a missing prefix are both quiet recall bugs, so neither
    is allowed to pass unnoticed.
    """
    if text.startswith(BGE_QUERY_PREFIX):
        raise RetrievalError(
            "query text already carries the bge prefix; it must be applied exactly once"
        )
    return BGE_QUERY_PREFIX + text


def _render_measurements(evidence: dict[str, object]) -> str:
    """Render the numeric part of a signal's evidence, sorted by key.

    **Only** ``int``, ``float`` and ``bool`` values are rendered.  String and
    list values are skipped -- that is the mechanical boundary that keeps
    capture-derived text such as ``top_parent_domain`` out of a query, and it
    is a structural rule rather than a list of fields to remember.
    """
    parts: list[str] = []
    for key in sorted(evidence):
        value = evidence[key]
        if isinstance(value, bool):
            parts.append(f"{key}={'true' if value else 'false'}")
        elif isinstance(value, int):
            parts.append(f"{key}={value}")
        elif isinstance(value, float):
            parts.append(f"{key}={format(value, '.6g')}")
    return "; ".join(parts)


def _assert_no_capture_text(text: str, label: str) -> None:
    """Refuse a query that contains an address or a domain name.

    This should be unreachable: templates are hand-written, summaries are
    generated from numbers by :mod:`ai.rag.signals`, and only numeric evidence
    is rendered.  It is here because "should be unreachable" is exactly the
    assumption a future edit breaks, and a hostname reaching a query is a
    prompt-injection channel rather than a cosmetic slip.
    """
    address = _IPV4_PATTERN.search(text)
    if address is not None:
        raise RetrievalError(
            f"query {label!r} contains an IP address ({address.group()}); "
            "capture-derived values must never enter a retrieval query"
        )
    domain = _DOMAIN_PATTERN.search(text)
    if domain is not None:
        raise RetrievalError(
            f"query {label!r} contains a domain-shaped token ({domain.group()}); "
            "capture-derived hostnames must never enter a retrieval query"
        )


def _signal_query(signal: Signal) -> SignalQuery:
    """Build the query for one signal.

    Three deterministic lines: the topic to retrieve, the observation that
    triggered it, and the numbers behind it.
    """
    lines = [
        f"Network security knowledge about: {QUERY_TEMPLATES[signal.signal_type]}.",
        f"Observed: {signal.summary}",
    ]
    measurements = _render_measurements(signal.evidence)
    if measurements:
        lines.append(f"Measurements: {measurements}")

    text = "\n".join(lines)
    label = f"signal:{signal.signal_type.value}"
    _assert_no_capture_text(text, label)

    return SignalQuery(
        label=label,
        signal_type=signal.signal_type,
        signal_id=signal.signal_id,
        text=text,
        embedding_text=apply_query_prefix(text),
    )


def _capture_query(report: SignalReport) -> SignalQuery:
    """Build the one capture-wide query.

    Carries the profile -- protocols, ports, verdicts -- which no individual
    signal describes, and lists the signal *types* that fired so the shape of
    the capture as a whole is retrievable.  Every value here is an integer or a
    vocabulary term; the capture name is deliberately absent, since a file name
    is neither knowledge nor safe to assume clean.
    """
    profile = report.profile
    protocols = ", ".join(f"{name} {count}"
                          for name, count in profile.protocol_distribution.items())
    ports = ", ".join(str(port) for port, _ in profile.top_destination_ports[:8])
    verdicts = ", ".join(f"{name} {count}"
                         for name, count in profile.verdict_distribution.items())
    fired = ", ".join(signal.signal_type.value.replace("_", " ")
                      for signal in report.signals)

    lines = [
        "Network security knowledge about: the overall shape of a captured traffic "
        "sample, protocol and port distribution, and what a deep packet inspection "
        "engine can conclude from flow metadata alone.",
        f"Flows: {report.flow_count}.",
    ]
    if protocols:
        lines.append(f"Protocols by flow count: {protocols}.")
    if ports:
        lines.append(f"Most frequent destination ports: {ports}.")
    if verdicts:
        lines.append(f"Engine verdicts: {verdicts}.")
    if fired:
        lines.append(f"Observations present: {fired}.")

    text = "\n".join(lines)
    _assert_no_capture_text(text, CAPTURE_QUERY_LABEL)

    return SignalQuery(
        label=CAPTURE_QUERY_LABEL,
        signal_type=None,
        signal_id=None,
        text=text,
        embedding_text=apply_query_prefix(text),
    )


def build_queries(
    report: SignalReport, config: RetrievalConfig | None = None
) -> tuple[SignalQuery, ...]:
    """Build every query for a signal report, in a fixed order.

    Pure and offline: no model, no store, no network.  One query per signal in
    the report's own order (severity, then confidence, then type), followed by
    the capture-wide query when it is enabled.  Separated from retrieval so
    query text can be inspected and tested with nothing installed.
    """
    cfg = config or RetrievalConfig()
    queries = [_signal_query(signal) for signal in report.signals]
    if cfg.include_capture_query and report.flow_count > 0:
        queries.append(_capture_query(report))
    return tuple(queries)


# ===========================================================================
# Models
# ===========================================================================
class SignalQuery(BaseModel):
    """One query: what is being asked, and which observation asked it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(description='"capture", or "signal:<signal_type>".')
    signal_type: SignalType | None = Field(
        default=None, description="None for the capture-wide query."
    )
    signal_id: str | None = Field(
        default=None, description="The signal this came from, for traceability."
    )
    text: str = Field(min_length=10, description="Query text, without the model prefix.")
    embedding_text: str = Field(
        min_length=10, description="Exactly BGE_QUERY_PREFIX + text; what gets embedded."
    )

    @field_validator("label")
    @classmethod
    def _label_shape(cls, v: str) -> str:
        if not _LABEL_PATTERN.match(v):
            raise ValueError(
                f'label {v!r} must be "capture" or "signal:<signal_type>"'
            )
        return v

    @model_validator(mode="after")
    def _consistent(self) -> SignalQuery:
        if self.embedding_text != BGE_QUERY_PREFIX + self.text:
            raise ValueError(
                "embedding_text must be the bge query prefix followed by text, "
                "applied exactly once"
            )
        if self.text.startswith(BGE_QUERY_PREFIX):
            raise ValueError("text must not already carry the bge query prefix")
        if self.signal_type is None:
            if self.label != CAPTURE_QUERY_LABEL:
                raise ValueError("only the capture query may have no signal type")
            if self.signal_id is not None:
                raise ValueError("the capture query has no signal id")
        else:
            if self.label != f"signal:{self.signal_type.value}":
                raise ValueError("label does not match signal_type")
            if not self.signal_id:
                raise ValueError("a signal query must name its signal id")
        return self


class RetrievedChunk(BaseModel):
    """One knowledge chunk that survived merging, with why it was retrieved.

    Composition, as in :class:`~ai.rag.vector_store.VectorRecord`: the chunk is
    held whole rather than having its title, section, category and text copied
    out.  The vector is deliberately *not* carried -- it is an index-internal
    detail, and a prompt builder needs the text and the provenance, not 384
    floats.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: KnowledgeChunk
    similarity: float = Field(
        ge=-1.0, le=1.0, description="Best cosine score across the queries that hit it."
    )
    rank: int = Field(ge=0, description="Zero-based position in the final result.")
    matched_signal_types: tuple[SignalType, ...] = Field(
        default=(), description="Signals whose query retrieved this chunk, sorted."
    )
    matched_query_labels: tuple[str, ...] = Field(
        min_length=1, description="Every query that retrieved this chunk, sorted."
    )
    per_query_similarity: dict[str, float] = Field(
        min_length=1, description="Query label -> score, for auditing the merge."
    )

    # -- validators ---------------------------------------------------------
    @field_validator("similarity")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("similarity must be a finite number")
        return v

    @field_validator("matched_query_labels")
    @classmethod
    def _labels_sorted(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError("matched_query_labels contains duplicates")
        if list(v) != sorted(v):
            raise ValueError("matched_query_labels must be sorted for determinism")
        for label in v:
            if not _LABEL_PATTERN.match(label):
                raise ValueError(f"matched_query_labels contains an invalid label: {label!r}")
        return v

    @field_validator("matched_signal_types")
    @classmethod
    def _types_sorted(cls, v: tuple[SignalType, ...]) -> tuple[SignalType, ...]:
        values = [item.value for item in v]
        if len(set(values)) != len(values):
            raise ValueError("matched_signal_types contains duplicates")
        if values != sorted(values):
            raise ValueError("matched_signal_types must be sorted for determinism")
        return v

    @model_validator(mode="after")
    def _merge_is_coherent(self) -> RetrievedChunk:
        if set(self.per_query_similarity) != set(self.matched_query_labels):
            raise ValueError(
                "per_query_similarity keys must be exactly the matched query labels"
            )
        for label, score in self.per_query_similarity.items():
            if score != score or score in (float("inf"), float("-inf")):
                raise ValueError(f"per_query_similarity[{label!r}] is not finite")
            if not -1.0 <= score <= 1.0:
                raise ValueError(f"per_query_similarity[{label!r}] is outside [-1, 1]")
        if abs(max(self.per_query_similarity.values()) - self.similarity) > 1e-12:
            raise ValueError(
                "similarity must be the best of the per-query scores"
            )
        expected = tuple(sorted(
            label.split(":", 1)[1] for label in self.matched_query_labels
            if label != CAPTURE_QUERY_LABEL
        ))
        if tuple(t.value for t in self.matched_signal_types) != expected:
            raise ValueError(
                "matched_signal_types must be exactly the signal labels that matched"
            )
        return self

    # -- provenance, read through to the chunk ------------------------------
    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def document_id(self) -> str:
        return self.chunk.document_id

    @property
    def title(self) -> str:
        return self.chunk.title

    @property
    def category(self) -> Category:
        return self.chunk.category

    @property
    def section(self) -> str:
        return self.chunk.section

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def heading_path(self) -> str:
        return self.chunk.heading_path

    @property
    def licence(self) -> str:
        return self.chunk.licence

    @property
    def sources(self) -> list[str]:
        return list(self.chunk.sources)

    def citation(self) -> str:
        return self.chunk.citation()

    def declares_matched_signal(self) -> bool:
        """Whether the source document lists any matching signal in ``applies_to``.

        Informational only -- it never affects ranking.  A chunk retrieved on
        similarity alone is a perfectly good hit; this simply records whether
        the corpus author also expected that connection.
        """
        declared = set(self.chunk.applies_to)
        return any(item.value in declared for item in self.matched_signal_types)

    def metadata(self) -> dict[str, object]:
        """Flat provenance view, without the chunk text."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "category": self.category.value,
            "section": self.section,
            "rank": self.rank,
            "similarity": self.similarity,
            "matched_signal_types": [item.value for item in self.matched_signal_types],
            "matched_query_labels": list(self.matched_query_labels),
            "licence": self.licence,
        }


class RetrievalReport(BaseModel):
    """Everything one retrieval run produced."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = RETRIEVAL_SCHEMA_VERSION
    generated_from: str = Field(description="The signal report this came from.")
    capture_name: str
    model_name: str = Field(description="Embedding model shared by query and index.")
    dimension: int = Field(ge=1)

    query_count: int = Field(ge=0)
    queries: tuple[SignalQuery, ...] = ()
    chunk_count: int = Field(ge=0)
    chunks: tuple[RetrievedChunk, ...] = ()
    parameters: dict[str, float | int | bool | str] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()

    # NOTE: no `retrieved_at`.  A timestamp would make two runs over identical
    # input differ and defeat every determinism guarantee here.

    @model_validator(mode="after")
    def _consistent(self) -> RetrievalReport:
        if self.query_count != len(self.queries):
            raise ValueError("query_count does not match the number of queries")
        if self.chunk_count != len(self.chunks):
            raise ValueError("chunk_count does not match the number of chunks")

        ids = [chunk.chunk_id for chunk in self.chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("the same chunk appears more than once in the result")

        if [chunk.rank for chunk in self.chunks] != list(range(len(self.chunks))):
            raise ValueError("ranks must be contiguous and start at zero")

        keys = [(-round(chunk.similarity, 12), chunk.chunk_id) for chunk in self.chunks]
        if keys != sorted(keys):
            raise ValueError(
                "chunks are not in the documented order "
                "(similarity descending, then chunk_id ascending)"
            )

        labels = {query.label for query in self.queries}
        for chunk in self.chunks:
            unknown = set(chunk.matched_query_labels) - labels
            if unknown:
                raise ValueError(
                    f"chunk {chunk.chunk_id} cites queries that were never run: "
                    f"{sorted(unknown)}"
                )
        return self

    # -- helpers ------------------------------------------------------------
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(chunk.chunk_id for chunk in self.chunks)

    def document_ids(self) -> tuple[str, ...]:
        """Distinct source documents, in result order."""
        seen: list[str] = []
        for chunk in self.chunks:
            if chunk.document_id not in seen:
                seen.append(chunk.document_id)
        return tuple(seen)

    def for_signal(self, signal_type: SignalType) -> tuple[RetrievedChunk, ...]:
        return tuple(chunk for chunk in self.chunks
                     if signal_type in chunk.matched_signal_types)

    def to_json(self, include_text: bool = False) -> str:
        """Stable JSON.  Chunk text is excluded by default -- it is long, and
        the interesting part of a retrieval run is which chunks came back."""
        payload = {
            "schema_version": self.schema_version,
            "generated_from": self.generated_from,
            "capture_name": self.capture_name,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "parameters": self.parameters,
            "notes": list(self.notes),
            "queries": [
                {"label": query.label, "text": query.text,
                 "signal_id": query.signal_id}
                for query in self.queries
            ],
            "chunks": [
                dict(chunk.metadata(), **({"text": chunk.text} if include_text else {}))
                for chunk in self.chunks
            ],
        }
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


# ===========================================================================
# Retrieval
# ===========================================================================
def _check_compatible(store: VectorStore, embedder: EmbeddingModel) -> None:
    """Refuse to search an index built with a different model."""
    if store.count() == 0:
        raise RetrievalError("the vector store is empty; build the index first")
    if store.model_name is not None and store.model_name != embedder.model_name:
        raise ModelMismatchError(
            f"the index was built with {store.model_name!r} but the query would be "
            f"embedded with {embedder.model_name!r}. Vectors from different models "
            "occupy different spaces, so their similarity is meaningless."
        )


def retrieve_for_signals(
    report: SignalReport,
    store: VectorStore,
    embedder: EmbeddingModel,
    config: RetrievalConfig | None = None,
) -> RetrievalReport:
    """Retrieve the knowledge chunks most relevant to a capture's signals.

    One query per signal plus one capture-wide query, each searched
    independently through :meth:`~ai.rag.vector_store.VectorStore.search`, then
    merged.

    **Merging.**  A chunk found by several queries appears once.  Its score is
    the *best* it achieved across them -- not a sum, which would reward a chunk
    for being vaguely related to everything, and not a mean, which would punish
    a chunk for being a precise answer to exactly one signal.  Every query that
    found it is recorded, so a later step can say *why* a chunk is present.

    **Ranking.**  Similarity descending, then ``chunk_id`` ascending -- the same
    rule :class:`~ai.rag.vector_store.VectorStore` uses, so the merged order is
    an extension of the per-query order rather than a second, different policy.

    **Diversity.**  At most ``max_per_document`` chunks from any one document,
    applied after ranking and before truncation, because a single long document
    would otherwise fill every slot.

    Deterministic throughout: no clock, no randomness, no unordered iteration.
    """
    cfg = config or RetrievalConfig()
    _check_compatible(store, embedder)

    queries = build_queries(report, cfg)
    notes: list[str] = []

    if not queries:
        notes.append("No signals fired and no capture query was built; nothing was searched.")
        return RetrievalReport(
            generated_from=f"ai.rag.signals.SignalReport v{report.schema_version}",
            capture_name=report.capture_name,
            model_name=embedder.model_name,
            dimension=store.dimension or 0,
            query_count=0, queries=(), chunk_count=0, chunks=(),
            parameters=cfg.as_dict(), notes=tuple(notes),
        )

    # One embedding call for every query, in order -- the model is loaded once
    # and reused, which is what EmbeddingModel already guarantees.
    vectors = embedder.embed_texts([query.embedding_text for query in queries])
    if len(vectors) != len(queries):  # pragma: no cover - embed_texts guarantees this
        raise RetrievalError("the embedder returned the wrong number of query vectors")

    if store.dimension is not None and len(vectors[0]) != store.dimension:
        raise ModelMismatchError(
            f"query vectors have {len(vectors[0])} dimensions but the index holds "
            f"{store.dimension}-dimension vectors"
        )

    # -- search each query independently ------------------------------------
    best: dict[str, float] = {}
    per_query: dict[str, dict[str, float]] = {}
    chunks: dict[str, KnowledgeChunk] = {}

    for query, vector in zip(queries, vectors):
        if cfg.per_query_top_k == 0:
            continue
        hits = store.search(vector, top_k=cfg.per_query_top_k,
                            min_similarity=cfg.min_similarity)
        for hit in hits:
            chunk_id = hit.chunk_id
            chunks.setdefault(chunk_id, hit.record.chunk)
            per_query.setdefault(chunk_id, {})[query.label] = hit.similarity
            if chunk_id not in best or hit.similarity > best[chunk_id]:
                best[chunk_id] = hit.similarity

    if not best:
        notes.append("No chunk met the retrieval criteria.")

    # -- rank, then cap per document, then truncate -------------------------
    ordered = sorted(best, key=lambda chunk_id: (-round(best[chunk_id], 12), chunk_id))

    selected: list[str] = []
    per_document: dict[str, int] = {}
    dropped_for_diversity = 0
    for chunk_id in ordered:
        if len(selected) >= cfg.final_top_k:
            break
        document_id = chunks[chunk_id].document_id
        if (cfg.max_per_document is not None
                and per_document.get(document_id, 0) >= cfg.max_per_document):
            dropped_for_diversity += 1
            continue
        per_document[document_id] = per_document.get(document_id, 0) + 1
        selected.append(chunk_id)

    if dropped_for_diversity:
        notes.append(
            f"{dropped_for_diversity} chunk(s) were dropped by the "
            f"max_per_document={cfg.max_per_document} diversity cap."
        )
    if len(ordered) > len(selected) + dropped_for_diversity:
        notes.append(
            f"{len(ordered)} chunk(s) matched; the top {len(selected)} are included."
        )

    results: list[RetrievedChunk] = []
    for rank, chunk_id in enumerate(selected):
        labels = tuple(sorted(per_query[chunk_id]))
        signal_types = tuple(
            SignalType(label.split(":", 1)[1])
            for label in labels if label != CAPTURE_QUERY_LABEL
        )
        results.append(
            RetrievedChunk(
                chunk=chunks[chunk_id],
                similarity=best[chunk_id],
                rank=rank,
                matched_signal_types=signal_types,
                matched_query_labels=labels,
                per_query_similarity={label: per_query[chunk_id][label] for label in labels},
            )
        )

    return RetrievalReport(
        generated_from=f"ai.rag.signals.SignalReport v{report.schema_version}",
        capture_name=report.capture_name,
        model_name=embedder.model_name,
        dimension=store.dimension or len(vectors[0]),
        query_count=len(queries),
        queries=queries,
        chunk_count=len(results),
        chunks=tuple(results),
        parameters=cfg.as_dict(),
        notes=tuple(notes),
    )


# ===========================================================================
# Manual check:  python -m ai.rag.retrieval
# ===========================================================================
if __name__ == "__main__":  # pragma: no cover - manual check
    import hashlib

    from ..schemas import CaptureReport, CaptureTotals, FlowRecord, TransportProtocol
    from .chunking import chunk_document
    from .documents import parse_document
    from .embeddings import EmbeddingConfig, EmbeddingResult
    from .signals import extract_signals
    from .vector_store import VectorRecord

    print("demo mode: four-dimension toy vectors, not real embeddings.")
    print("They exist to show the retrieval mechanics with no model installed;")
    print("nothing here says anything about semantic quality.\n")

    class _ToyEncoder:
        """Demo-only encoder: a fixed keyword axis, so hits are explainable.

        Lives inside ``__main__`` and is passed in explicitly.  The library
        never constructs one -- there is no fallback encoder anywhere in
        :mod:`ai.rag.embeddings`.
        """

        AXES = ("dns", "http", "scan", "browsing")

        @property
        def name(self) -> str:
            return "demo/toy-4d"

        def dimension(self) -> int:
            return 4

        def encode(self, texts, normalize: bool):
            rows = []
            for text in texts:
                lowered = text.lower()
                row = [1.0 + lowered.count(axis) for axis in self.AXES]
                if normalize:
                    length = sum(value * value for value in row) ** 0.5
                    row = [value / length for value in row]
                rows.append(row)
            return rows

    DEMO_DOC = """\
---
id: demo-dns
title: Demo DNS Notes
category: protocols
version: 1.0
updated: 2026-08-27
applies_to:
  - dns_high_volume
keywords:
  - dns
mitre: []
severity_hint: info
sources:
  - Authored for this project.
licence: project-authored
---

## Summary

DNS resolution turns names into addresses and is the noisiest protocol present.

## What the DPI engine can observe

The `protocol`, `dst_port` and `server_name` fields of each `FlowRecord`.

## Indicators

Many DNS queries, high name cardinality, and long labels.

## Benign explanations

Browsing a modern page resolves dozens of names across many hosts.

## Recommended checks

Compare distinct DNS names against DNS flow count.

## References

See the protocols and baselines categories.
"""

    demo_config = EmbeddingConfig(model_name="demo/toy-4d")
    demo_embedder = EmbeddingModel(demo_config, encoder=_ToyEncoder())

    demo_chunks = chunk_document(parse_document(DEMO_DOC, "protocols/demo-dns.md"))
    demo_store = VectorStore("demo")
    demo_vectors = demo_embedder.embed_texts(
        [f"{chunk.heading_path}\n\n{chunk.text}" for chunk in demo_chunks]
    )
    for demo_chunk, demo_vector in zip(demo_chunks, demo_vectors):
        demo_store.add(VectorRecord(
            chunk=demo_chunk,
            embedding=EmbeddingResult(
                chunk_id=demo_chunk.chunk_id,
                document_id=demo_chunk.document_id,
                category=demo_chunk.category,
                section=demo_chunk.section,
                heading_path=demo_chunk.heading_path,
                content_sha256=demo_chunk.content_sha256,
                input_sha256=hashlib.sha256(
                    demo_chunk.chunk_id.encode("utf-8")).hexdigest(),
                model_name="demo/toy-4d",
                dimension=4,
                normalized=True,
                vector=demo_vector,
            ),
        ))

    def demo_flow(flow_id: int, **overrides) -> FlowRecord:
        base = dict(
            flow_id=flow_id, protocol=TransportProtocol.UDP, dst_port=53, src_port=50000,
            server_name=f"name{flow_id}.example.test", application="DNS",
            state="CLASSIFIED", verdict="FORWARD", packets_out=1, packets_in=1,
            bytes_out=80, bytes_in=180, syn_seen=False, syn_ack_seen=False,
            fin_seen=False, src_ip="host-1", dst_ip="net-1",
        )
        base.update(overrides)
        return FlowRecord(**base)

    demo_flows = [demo_flow(i) for i in range(8)]
    demo_capture = CaptureReport(
        capture_name="demo.pcap",
        totals=CaptureTotals(
            total_packets=16, total_bytes=2080, tcp_packets=0, udp_packets=16,
            forwarded_packets=16, dropped_packets=0, total_flows=8, flows_included=8,
        ),
        application_distribution={"DNS": 8},
        top_server_names=[],
        blocking_rules_active={},
        flows=demo_flows,
        redaction_mode="redact_private",
        notes=[],
    )

    demo_signals = extract_signals(demo_capture)
    demo_result = retrieve_for_signals(demo_signals, demo_store, demo_embedder,
                                       RetrievalConfig(per_query_top_k=3, final_top_k=4,
                                                       max_per_document=None))

    print(f"capture:  {demo_result.capture_name}")
    print(f"signals:  {', '.join(demo_signals.types())}")
    print(f"queries:  {demo_result.query_count}")
    for demo_query in demo_result.queries:
        print(f"  {demo_query.label:<32} {demo_query.text.splitlines()[0][:64]}...")
    print(f"\nretrieved {demo_result.chunk_count} chunk(s):")
    for hit in demo_result.chunks:
        matched = ", ".join(hit.matched_query_labels)
        print(f"  #{hit.rank}  {hit.similarity:.4f}  {hit.citation()}")
        print(f"        matched by: {matched}")
    print(f"\nnotes: {'; '.join(demo_result.notes) or 'none'}")
