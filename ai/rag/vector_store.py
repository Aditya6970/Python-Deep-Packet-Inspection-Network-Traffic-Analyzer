"""A small, explicit, in-memory vector store with cosine similarity search.

What this does
--------------
Holds ``(chunk, embedding)`` pairs and answers one question: *which stored
vectors are most similar to this query vector?*  Similarity is cosine, computed
here in NumPy -- there is no vector database involved, and none is needed.

What this does **not** do: load documents, chunk them, call an embedding model,
build queries, decide what to retrieve, or talk to an LLM.  It accepts vectors
that already exist and returns ranked matches.  The pipeline's four
responsibilities stay in four modules::

    documents.py    text  -> validated documents
    chunking.py     documents -> chunks
    embeddings.py   chunks -> vectors
    vector_store.py vectors -> searchable index      <- this module

Nothing here imports the embedding layer, so the store can be exercised end to
end with hand-built vectors and no model on disk.

Why brute force, and why that is the right answer here
------------------------------------------------------
The corpus is 37 chunks of 384 dimensions: a 114 KB matrix.  One
``matrix @ query`` scores every record in well under a millisecond.  FAISS,
Chroma and friends exist to make *approximate* search fast at 10^6 vectors and
beyond; at 10^2 they would add a dependency, a build step and an opaque layer
in exchange for making an already-instant exact search slightly less exact.

Brute force is also honest about what it is.  Every number in a result can be
traced to two lines of arithmetic, which matters when the point of the exercise
is to understand retrieval rather than to import it.  The seam is small enough
that swapping in an ANN index later means replacing one method.

Precision
---------
Vectors are stored as ``float64``.  The embedding model emits ``float32``, so
this widens them -- deliberately: 37 x 384 x 8 bytes is 114 KB, precision is
free at this scale, and it keeps the cosine identity tight enough to assert in
tests.  ``float32`` storage is the obvious switch if the corpus ever grows into
the hundreds of thousands of chunks; it halves memory and costs a few decimal
places that ranking would never notice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Iterable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .chunking import KnowledgeChunk
from .documents import Category
from .embeddings import EmbeddingResult

__all__ = [
    "COSINE_TOLERANCE",
    "DuplicateChunkError",
    "InvalidVectorError",
    "SearchResult",
    "VectorRecord",
    "VectorStore",
    "VectorStoreError",
    "cosine_similarity",
]

#: How close a normalised vector's cosine score must be to its plain dot
#: product for the two to be considered equivalent.  Used by tests and by the
#: documentation of :meth:`VectorStore.search`, never by the search itself --
#: search always divides by the norms, so it needs no tolerance.
COSINE_TOLERANCE: Final[float] = 1e-9

#: Vectors shorter than one component, or with a zero norm, cannot have a
#: direction, and cosine similarity is undefined for them.
_ZERO_NORM_EPSILON: Final[float] = 1e-12


# ===========================================================================
# Errors
# ===========================================================================
class VectorStoreError(Exception):
    """Base class for vector store problems."""


class DuplicateChunkError(VectorStoreError):
    """A chunk id already present in the store was added again.

    Never resolved by overwriting.  Two records under one id would make search
    results depend on insertion order, and an index rebuilt from a changed
    corpus should be a fresh store, not a store quietly mutated in place.
    """


class InvalidVectorError(VectorStoreError):
    """A vector is empty, the wrong width, zero, or not finite."""


# ===========================================================================
# Records
# ===========================================================================
class VectorRecord(BaseModel):
    """One indexed item: a chunk paired with its embedding.

    Composition rather than duplication.  A flat record would restate
    ``title``, ``category``, ``section``, ``text``, ``model_name``,
    ``dimension`` and ``normalized`` -- every one of which already exists on
    :class:`~ai.rag.chunking.KnowledgeChunk` or
    :class:`~ai.rag.embeddings.EmbeddingResult`, each with its own validation.
    Copying them would create two places for each fact to live and one more
    way for them to disagree.

    Holding both objects instead gives every required field through a property,
    keeps a single source of truth for each, and makes one whole class of bug
    impossible: the validator below refuses a record whose embedding was
    computed for a *different chunk*, which a flat record could never detect.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: KnowledgeChunk
    embedding: EmbeddingResult

    @model_validator(mode="after")
    def _same_chunk(self) -> VectorRecord:
        if self.chunk.chunk_id != self.embedding.chunk_id:
            raise ValueError(
                f"embedding belongs to chunk {self.embedding.chunk_id!r} but was "
                f"paired with chunk {self.chunk.chunk_id!r}"
            )
        if self.chunk.content_sha256 != self.embedding.content_sha256:
            raise ValueError(
                f"embedding for {self.chunk.chunk_id!r} was computed from different "
                "chunk content; the corpus changed since it was embedded"
            )
        return self

    # -- provenance, read through to the source objects ---------------------
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

    # -- embedding ----------------------------------------------------------
    @property
    def vector(self) -> tuple[float, ...]:
        return self.embedding.vector

    @property
    def dimension(self) -> int:
        return self.embedding.dimension

    @property
    def model_name(self) -> str:
        return self.embedding.model_name

    @property
    def normalized(self) -> bool:
        return self.embedding.normalized

    def citation(self) -> str:
        """Human-readable provenance: which document and section this came from."""
        return self.chunk.citation()

    def metadata(self) -> dict[str, object]:
        """Flat provenance view for logs and exports.  Never includes the vector."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "category": self.category.value,
            "section": self.section,
            "heading_path": self.heading_path,
            "chunk_index": self.chunk.chunk_index,
            "chunk_count": self.chunk.chunk_count,
            "char_count": self.chunk.char_count,
            "licence": self.licence,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "normalized": self.normalized,
            "content_sha256": self.chunk.content_sha256,
        }


class SearchResult(BaseModel):
    """One ranked match."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record: VectorRecord
    similarity: float = Field(
        ge=-1.0 - 1e-6, le=1.0 + 1e-6,
        description="Cosine similarity with the query, in [-1, 1].",
    )
    rank: int = Field(ge=0, description="Zero-based position in the result list.")

    # -- convenience --------------------------------------------------------
    @property
    def chunk_id(self) -> str:
        return self.record.chunk_id

    @property
    def document_id(self) -> str:
        return self.record.document_id

    @property
    def section(self) -> str:
        return self.record.section

    @property
    def text(self) -> str:
        return self.record.text

    def citation(self) -> str:
        return self.record.citation()

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<SearchResult #{self.rank} {self.similarity:.4f} {self.chunk_id}>"


# ===========================================================================
# Cosine similarity
# ===========================================================================
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors.

    Written out rather than imported, because it is the whole idea::

        cos(a, b) = (a . b) / (||a|| * ||b||)

    Correct for vectors of any magnitude.  When both are L2-normalised the
    denominator is 1 and this reduces to the dot product -- which is the reason
    the embedding layer normalises by default, and the identity the tests
    assert to within :data:`COSINE_TOLERANCE`.

    Raises :class:`InvalidVectorError` for a zero vector: it has no direction,
    so the angle to it is undefined.  Returning 0.0 instead would be a silent
    lie that ranks a meaningless vector above genuinely dissimilar ones.
    """
    left = _as_vector(a, "a")
    right = _as_vector(b, "b")
    if left.shape != right.shape:
        raise InvalidVectorError(
            f"cannot compare a {left.shape[0]}-dimension vector with a "
            f"{right.shape[0]}-dimension one"
        )

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= _ZERO_NORM_EPSILON or right_norm <= _ZERO_NORM_EPSILON:
        raise InvalidVectorError("cosine similarity is undefined for a zero vector")

    return float(np.dot(left, right) / (left_norm * right_norm))


def _as_vector(values: Sequence[float], label: str) -> np.ndarray:
    """Convert to a validated 1-D float64 array."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidVectorError(f"{label} is not a numeric vector: {exc}") from exc

    if array.ndim != 1:
        raise InvalidVectorError(
            f"{label} must be a one-dimensional vector, got shape {array.shape}"
        )
    if array.size == 0:
        raise InvalidVectorError(f"{label} is empty; a vector needs at least one component")
    if not np.all(np.isfinite(array)):
        raise InvalidVectorError(f"{label} contains NaN or infinity")
    return array


# ===========================================================================
# The store
# ===========================================================================
class VectorStore:
    """An in-memory index of :class:`VectorRecord` objects.

    Vectors live in a list of frozen 1-D arrays and are stacked into one
    ``N x D`` matrix the first time a search needs it.  The matrix and the row
    norms are cached and invalidated on every mutation, so repeated searches
    over an unchanged store do no work beyond the multiply.

    Homogeneity is enforced on the way in: every vector must share the first
    one's width **and** its model name.  Mixing two embedding models in one
    index produces vectors from two different spaces whose cosine scores are
    meaningless -- and nothing downstream would ever notice, which is exactly
    why it is rejected here.
    """

    __slots__ = ("_name", "_records", "_vectors", "_index", "_matrix", "_norms",
                 "_dimension", "_model_name")

    def __init__(self, name: str = "knowledge") -> None:
        self._name = name
        self._records: list[VectorRecord] = []
        self._vectors: list[np.ndarray] = []
        self._index: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        self._norms: np.ndarray | None = None
        self._dimension: int | None = None
        self._model_name: str | None = None

    # -- introspection ------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._index

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"<VectorStore {self._name!r} records={len(self._records)} "
                f"dim={self._dimension} model={self._model_name!r}>")

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int | None:
        """Vector width of this store, or ``None`` while it is empty."""
        return self._dimension

    @property
    def model_name(self) -> str | None:
        """The one embedding model this store holds, or ``None`` while empty."""
        return self._model_name

    def count(self) -> int:
        return len(self._records)

    def chunk_ids(self) -> tuple[str, ...]:
        """Ids in insertion order."""
        return tuple(record.chunk_id for record in self._records)

    def records(self) -> tuple[VectorRecord, ...]:
        """All records, in insertion order.  Records are frozen models."""
        return tuple(self._records)

    def get(self, chunk_id: str) -> VectorRecord | None:
        position = self._index.get(chunk_id)
        return None if position is None else self._records[position]

    def matrix(self) -> np.ndarray:
        """A **copy** of the ``N x D`` matrix, for inspection and testing.

        A copy on purpose: handing out the live array would let a caller
        rewrite the index by accident.  The internal rows are additionally
        marked read-only, so even an internal mistake raises rather than
        silently corrupting a vector.
        """
        if not self._records:
            return np.empty((0, self._dimension or 0), dtype=np.float64)
        return self._build_matrix().copy()

    # -- mutation -----------------------------------------------------------
    def add(self, record: VectorRecord) -> None:
        """Add one record.  Raises rather than overwriting an existing id."""
        self._validate(record)
        self._insert(record)
        self._invalidate()

    def add_many(self, records: Iterable[VectorRecord]) -> int:
        """Add several records atomically, returning how many were added.

        Every record is validated *before* any is inserted, so a bad record in
        the middle of a batch leaves the store exactly as it was.  A partially
        populated index is worse than a rejected one: it looks fine and
        retrieves incompletely.
        """
        incoming = list(records)
        if not incoming:
            return 0

        seen: set[str] = set()
        dimension = self._dimension
        model = self._model_name
        for record in incoming:
            self._validate(record, dimension=dimension, model_name=model)
            if record.chunk_id in seen:
                raise DuplicateChunkError(
                    f"chunk id {record.chunk_id!r} appears twice in the same batch"
                )
            seen.add(record.chunk_id)
            dimension = record.dimension
            model = record.model_name

        for record in incoming:
            self._insert(record)
        self._invalidate()
        return len(incoming)

    def clear(self) -> None:
        """Empty the store, including its dimension and model commitments."""
        self._records.clear()
        self._vectors.clear()
        self._index.clear()
        self._dimension = None
        self._model_name = None
        self._invalidate()

    # -- search -------------------------------------------------------------
    def search(
        self,
        query: Sequence[float],
        top_k: int = 5,
        min_similarity: float | None = None,
    ) -> tuple[SearchResult, ...]:
        """Return the ``top_k`` records most similar to ``query``.

        The query is validated (width, finiteness, non-zero), scored against
        every stored vector in one matrix multiply, filtered by
        ``min_similarity`` if given, then ranked.

        Cosine is computed in full -- ``dot / (row_norm * query_norm)`` -- so
        the result is correct whether or not the stored vectors are
        normalised.  For normalised vectors the denominator is 1 and the score
        equals the plain dot product, which is why normalising at embedding
        time is worth doing and why this is safe to rely on later.

        ``top_k=0`` returns an empty tuple: a legitimate request for nothing,
        not an error.  A negative ``top_k`` is a caller bug and raises.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError(f"top_k must be an integer, got {type(top_k).__name__}")
        if top_k < 0:
            raise ValueError(f"top_k must be zero or positive, got {top_k}")
        if min_similarity is not None and not -1.0 <= float(min_similarity) <= 1.0:
            raise ValueError(
                f"min_similarity must be within [-1, 1], got {min_similarity}"
            )

        vector = _as_vector(query, "query")
        if self._dimension is not None and vector.shape[0] != self._dimension:
            raise InvalidVectorError(
                f"query has {vector.shape[0]} dimensions but this store holds "
                f"{self._dimension}-dimension vectors"
            )
        query_norm = float(np.linalg.norm(vector))
        if query_norm <= _ZERO_NORM_EPSILON:
            raise InvalidVectorError("query is a zero vector; its direction is undefined")

        if not self._records or top_k == 0:
            return ()

        matrix = self._build_matrix()
        norms = self._build_norms()

        # N scores in one pass:  (N x D) @ (D,) -> (N,), then divide by norms.
        scores = (matrix @ vector) / (norms * query_norm)

        # Deterministic ranking: similarity descending, then chunk_id ascending.
        # Ties are exact float equality -- two scores differing in their last
        # bits are ordered by value, not by id, which is the honest behaviour.
        order = sorted(
            range(len(self._records)),
            key=lambda position: (-float(scores[position]),
                                  self._records[position].chunk_id),
        )

        results: list[SearchResult] = []
        for position in order:
            score = float(scores[position])
            if min_similarity is not None and score < min_similarity:
                continue
            results.append(
                SearchResult(
                    record=self._records[position],
                    similarity=_clamp(score),
                    rank=len(results),
                )
            )
            if len(results) == top_k:
                break

        return tuple(results)

    # -- export -------------------------------------------------------------
    def to_json(self, include_vectors: bool = False) -> str:
        """Deterministic JSON dump, for debugging and diffing index builds.

        Sorted by ``chunk_id`` with sorted keys, so two dumps of equivalent
        stores compare byte for byte regardless of insertion order.  Vectors
        are excluded by default -- 37 x 384 floats is unreadable noise, and the
        metadata is what a human actually wants to inspect.

        This is *not* persistence.  There is no loader, no database and no
        format guarantee; a real index format belongs with the build step.
        """
        payload = {
            "name": self._name,
            "count": len(self._records),
            "dimension": self._dimension,
            "model_name": self._model_name,
            "records": [
                dict(record.metadata(),
                     **({"vector": [float(x) for x in record.vector]}
                        if include_vectors else {}))
                for record in sorted(self._records, key=lambda r: r.chunk_id)
            ],
        }
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)

    def export_json(self, path: str | Path, include_vectors: bool = False) -> Path:
        """Write :meth:`to_json` to ``path`` and return it."""
        destination = Path(path)
        destination.write_text(self.to_json(include_vectors), encoding="utf-8")
        return destination

    # -- internals ----------------------------------------------------------
    def _validate(
        self,
        record: VectorRecord,
        dimension: int | None = None,
        model_name: str | None = None,
    ) -> None:
        """Check one record against the store's commitments.

        ``dimension``/``model_name`` may be supplied so :meth:`add_many` can
        validate a whole batch against the state it *will* have, before
        inserting anything.
        """
        if not isinstance(record, VectorRecord):
            raise VectorStoreError(
                f"expected a VectorRecord, got {type(record).__name__}"
            )

        expected_dimension = self._dimension if dimension is None else dimension
        expected_model = self._model_name if model_name is None else model_name

        if record.chunk_id in self._index:
            raise DuplicateChunkError(
                f"chunk id {record.chunk_id!r} is already in this store; "
                "the store never overwrites -- clear it or build a new one"
            )

        vector = _as_vector(record.vector, f"vector for {record.chunk_id}")
        if vector.shape[0] != record.dimension:
            raise InvalidVectorError(
                f"{record.chunk_id}: vector has {vector.shape[0]} components but "
                f"its embedding reports dimension {record.dimension}"
            )
        if float(np.linalg.norm(vector)) <= _ZERO_NORM_EPSILON:
            raise InvalidVectorError(
                f"{record.chunk_id}: vector is all zeros, so it has no direction "
                "and cosine similarity against it is undefined"
            )

        if expected_dimension is not None and vector.shape[0] != expected_dimension:
            raise InvalidVectorError(
                f"{record.chunk_id}: this store holds {expected_dimension}-dimension "
                f"vectors, cannot add a {vector.shape[0]}-dimension one"
            )
        if expected_model is not None and record.model_name != expected_model:
            raise InvalidVectorError(
                f"{record.chunk_id}: this store holds vectors from "
                f"{expected_model!r}, cannot mix in {record.model_name!r}. "
                "Vectors from different models occupy different spaces and their "
                "similarity scores are meaningless."
            )

    def _insert(self, record: VectorRecord) -> None:
        vector = _as_vector(record.vector, record.chunk_id)
        vector.flags.writeable = False  # protect the stored copy from mutation
        self._index[record.chunk_id] = len(self._records)
        self._records.append(record)
        self._vectors.append(vector)
        self._dimension = vector.shape[0]
        self._model_name = record.model_name

    def _invalidate(self) -> None:
        self._matrix = None
        self._norms = None

    def _build_matrix(self) -> np.ndarray:
        if self._matrix is None:
            matrix = np.vstack(self._vectors) if self._vectors else np.empty(
                (0, self._dimension or 0), dtype=np.float64
            )
            matrix.flags.writeable = False
            self._matrix = matrix
        return self._matrix

    def _build_norms(self) -> np.ndarray:
        if self._norms is None:
            norms = np.linalg.norm(self._build_matrix(), axis=1)
            norms.flags.writeable = False
            self._norms = norms
        return self._norms


def _clamp(score: float) -> float:
    """Pull a score back inside [-1, 1].

    Floating-point division can leave a self-comparison at 1.0000000000000002,
    which is not a real similarity above one -- it is the last bit of a
    rounding error, and clamping keeps the contract exact.
    """
    return max(-1.0, min(1.0, score))


# ===========================================================================
# Manual check:  python -m ai.rag.vector_store
# ===========================================================================
if __name__ == "__main__":  # pragma: no cover - manual check
    # The demo builds its own chunks so the store can be shown working with no
    # embedding model installed.  These imports live here, not at module
    # scope: the store itself must not know how documents are loaded.
    import hashlib

    from .chunking import chunk_document
    from .documents import parse_document

    DEMO = """\
---
id: demo-document
title: Demo Document
category: protocols
version: 1.0
updated: 2026-08-27
applies_to:
  - dns_high_volume
keywords:
  - demo
mitre: []
severity_hint: info
sources:
  - Authored for this project.
licence: project-authored
---

## Summary

A tiny document used only to demonstrate the vector store.

## What the DPI engine can observe

The `server_name`, `dst_port` and `protocol` fields of each `FlowRecord`.

## Indicators

Many DNS queries with long random-looking names under one parent domain.

## Benign explanations

Security vendors encode file hashes into DNS lookups by design.

## Recommended checks

Group queries by parent domain and compare distinct names to flow count.

## References

See dns-normal-behaviour and dns-tunneling.
"""

    # Four-dimension toy vectors, chosen by hand so the ranking is obvious.
    TOY_VECTORS = {
        "Summary": (1.0, 0.0, 0.0, 0.0),
        "What the DPI engine can observe": (0.9, 0.43589, 0.0, 0.0),
        "Indicators": (0.0, 1.0, 0.0, 0.0),
        "Benign explanations": (0.0, 0.0, 1.0, 0.0),
        "Recommended checks": (0.0, 0.0, 0.0, 1.0),
        "References": (0.5, 0.5, 0.5, 0.5),
    }

    demo_chunks = chunk_document(parse_document(DEMO, "protocols/demo-document.md"))
    store = VectorStore("demo")
    for demo_chunk in demo_chunks:
        raw = TOY_VECTORS[demo_chunk.section]
        length = sum(component * component for component in raw) ** 0.5
        unit = tuple(component / length for component in raw)
        store.add(
            VectorRecord(
                chunk=demo_chunk,
                embedding=EmbeddingResult(
                    chunk_id=demo_chunk.chunk_id,
                    document_id=demo_chunk.document_id,
                    category=demo_chunk.category,
                    section=demo_chunk.section,
                    heading_path=demo_chunk.heading_path,
                    content_sha256=demo_chunk.content_sha256,
                    input_sha256=hashlib.sha256(
                        demo_chunk.chunk_id.encode("utf-8")
                    ).hexdigest(),
                    model_name="demo/toy-4d",
                    dimension=len(unit),
                    normalized=True,
                    vector=unit,
                ),
            )
        )

    print(f"store:     {store.name}")
    print(f"records:   {store.count()}")
    print(f"dimension: {store.dimension}")
    print(f"model:     {store.model_name}")
    print(f"matrix:    {store.matrix().shape[0]} x {store.matrix().shape[1]}")

    demo_query = (1.0, 0.0, 0.0, 0.0)
    print(f"\nquery {demo_query} -> top 3:")
    for hit in store.search(demo_query, top_k=3):
        print(f"  #{hit.rank}  {hit.similarity:+.4f}  {hit.citation()}")

    print("\nwith min_similarity=0.8:")
    for hit in store.search(demo_query, top_k=5, min_similarity=0.8):
        print(f"  #{hit.rank}  {hit.similarity:+.4f}  {hit.citation()}")
