"""Test runner for the local vector store -- RAG step 4 only.

Two tiers
---------
**Unit tier** -- always runs.  Everything about the store: validation, cosine
arithmetic, ranking, tie-breaking, thresholds, mutation protection, export.  It
uses hand-built four-dimension vectors over a synthetic document parsed in
memory, so it needs **no embedding model, no downloads and no network** -- only
numpy and pydantic.  Because the vectors are chosen by hand, every expected
score below can be checked with a pencil.

**Integration tier** -- runs only when ``BAAI/bge-small-en-v1.5`` can be
loaded.  It embeds the real corpus **once**, indexes it and checks the shape of
the result.  When the model is unavailable those checks SKIP with the reason
and the suite still passes.

Two of the store's guards cannot be reached through normal construction,
because :class:`~ai.rag.embeddings.EmbeddingResult` already rejects empty,
NaN and infinite vectors.  Those cases are built with
``model_construct`` -- pydantic's validation bypass -- specifically to prove
the store validates independently rather than trusting its input.

Run::

    python run_rag_vector_store_tests.py
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import sys
import tempfile
from pathlib import Path

import numpy as np

# Import the package so its console-encoding fix is applied before any output.
import dpi  # noqa: F401
from ai.rag.chunking import KnowledgeChunk, chunk_corpus, chunk_document
from ai.rag.documents import load_corpus, parse_document
from ai.rag.embeddings import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    EmbeddingConfig,
    EmbeddingModel,
    EmbeddingResult,
    ModelUnavailableError,
    sentence_transformers_available,
)
from ai.rag.vector_store import (
    COSINE_TOLERANCE,
    DuplicateChunkError,
    InvalidVectorError,
    SearchResult,
    VectorRecord,
    VectorStore,
    VectorStoreError,
    cosine_similarity,
)

_passed = 0
_failed = 0
_skipped = 0

MODEL_NAME = "test/toy-4d"
DIMENSION = 4


def check(label: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}" + (f"  -- {detail}" if detail else ""))


def skip(label: str, why: str) -> None:
    global _skipped
    _skipped += 1
    print(f"  SKIP  {label}  -- {why}")


def raises(label: str, expected: type[Exception], call) -> None:
    """Assert ``call()`` raises ``expected`` with a message worth reading."""
    try:
        call()
    except expected as exc:
        check(label, len(str(exc)) > 15, f"error message too terse: {str(exc)!r}")
    except Exception as exc:  # noqa: BLE001 - wrong type is the failure
        check(label, False, f"raised {type(exc).__name__}: {exc}")
    else:
        check(label, False, f"no {expected.__name__} was raised")


# ===========================================================================
# Synthetic fixtures -- no embedding model involved
# ===========================================================================
DOCUMENT = """\
---
id: {doc_id}
title: {title}
category: protocols
version: 1.0
updated: 2026-08-27
applies_to:
  - dns_high_volume
keywords:
  - fixture
mitre: []
severity_hint: info
sources:
  - Authored for this project.
licence: project-authored
---

## Summary

A synthetic document used to build chunks without touching the real corpus.

## What the DPI engine can observe

The `server_name`, `dst_port`, `protocol` and `bytes_out` fields of a `FlowRecord`.

## Indicators

Many DNS queries with long random-looking labels under a single parent domain.

## Benign explanations

Security vendors encode file hashes into DNS lookups by design.

## Recommended checks

Group queries by parent domain and compare distinct names against flow count.

## References

See dns-normal-behaviour and dns-tunneling.
"""


def fixture_chunks(doc_id: str = "fixture-document",
                   title: str = "Fixture Document") -> tuple[KnowledgeChunk, ...]:
    """Six chunks parsed and chunked entirely in memory."""
    text = DOCUMENT.format(doc_id=doc_id, title=title)
    document = parse_document(text, f"protocols/{doc_id}.md")
    return chunk_document(document)


def make_record(
    chunk: KnowledgeChunk,
    vector,
    *,
    model_name: str = MODEL_NAME,
    normalized: bool | None = None,
    validate: bool = True,
) -> VectorRecord:
    """Pair a chunk with a hand-built vector.

    ``validate=False`` bypasses :class:`EmbeddingResult` validation via
    ``model_construct``, which is the only way to hand the store a vector the
    embedding layer would have rejected -- exactly what the defence-in-depth
    checks need.
    """
    components = tuple(float(x) for x in vector)
    if normalized is None:
        length = math.sqrt(sum(c * c for c in components))
        normalized = abs(length - 1.0) < 1e-6

    fields = {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "category": chunk.category,
        "section": chunk.section,
        "heading_path": chunk.heading_path,
        "content_sha256": chunk.content_sha256,
        "input_sha256": hashlib.sha256(chunk.chunk_id.encode("utf-8")).hexdigest(),
        "model_name": model_name,
        "dimension": len(components),
        "normalized": normalized,
        "vector": components,
    }
    embedding = (EmbeddingResult(**fields) if validate
                 else EmbeddingResult.model_construct(**fields))
    return VectorRecord(chunk=chunk, embedding=embedding)


def unit(*components: float) -> tuple[float, ...]:
    """L2-normalise a hand-written vector."""
    length = math.sqrt(sum(c * c for c in components))
    return tuple(c / length for c in components)


#: Six orthogonal-ish unit vectors, one per section, chosen so the expected
#: ranking for any basis query can be worked out by hand.
BASIS = {
    "Summary": unit(1, 0, 0, 0),
    "What the DPI engine can observe": unit(1, 1, 0, 0),
    "Indicators": unit(0, 1, 0, 0),
    "Benign explanations": unit(0, 0, 1, 0),
    "Recommended checks": unit(0, 0, 0, 1),
    "References": unit(1, 1, 1, 1),
}


def populated_store(name: str = "fixture") -> tuple[VectorStore, tuple[KnowledgeChunk, ...]]:
    chunks = fixture_chunks()
    store = VectorStore(name)
    store.add_many([make_record(c, BASIS[c.section]) for c in chunks])
    return store, chunks


# ===========================================================================
# 1. Empty store
# ===========================================================================
def test_empty_store() -> None:
    print("\nEmpty store")

    store = VectorStore()
    check("an empty store is valid", store.count() == 0)
    check("len() agrees with count()", len(store) == store.count() == 0)
    check("an empty store has no dimension yet", store.dimension is None)
    check("an empty store has no model commitment yet", store.model_name is None)
    check("an empty store lists no chunk ids", store.chunk_ids() == ())
    check("an empty store contains nothing", "anything" not in store)
    check("get() on an empty store returns None", store.get("anything") is None)
    check("an empty store's matrix has zero rows", store.matrix().shape[0] == 0)
    check("searching an empty store returns no results",
          store.search((1.0, 0.0, 0.0, 0.0), top_k=5) == ())
    check("an empty store still exports valid JSON",
          json.loads(store.to_json())["count"] == 0)


# ===========================================================================
# 2. Adding
# ===========================================================================
def test_adding() -> None:
    print("\nAdding records")

    chunks = fixture_chunks()
    store = VectorStore()

    store.add(make_record(chunks[0], BASIS[chunks[0].section]))
    check("a single record can be added", store.count() == 1)
    check("the store adopts the record's dimension", store.dimension == DIMENSION)
    check("the store adopts the record's model", store.model_name == MODEL_NAME)
    check("the record can be looked up by chunk id",
          store.get(chunks[0].chunk_id) is not None)
    check("membership testing works", chunks[0].chunk_id in store)

    for chunk in chunks[1:]:
        store.add(make_record(chunk, BASIS[chunk.section]))
    check("multiple records can be added", store.count() == len(chunks))
    check("count() is correct after several adds", store.count() == 6)
    check("insertion order is preserved",
          store.chunk_ids() == tuple(c.chunk_id for c in chunks))

    # -- batch ------------------------------------------------------------
    batch_store = VectorStore()
    added = batch_store.add_many([make_record(c, BASIS[c.section]) for c in chunks])
    check("add_many reports how many it added", added == len(chunks))
    check("batch insertion produces the same store",
          batch_store.chunk_ids() == store.chunk_ids())
    check("add_many with an empty iterable is a no-op",
          batch_store.add_many([]) == 0 and batch_store.count() == len(chunks))

    # -- atomicity --------------------------------------------------------
    other = fixture_chunks("second-document", "Second Document")
    partial = VectorStore()
    partial.add(make_record(chunks[0], BASIS[chunks[0].section]))
    bad_batch = [
        make_record(other[1], BASIS[other[1].section]),
        make_record(other[2], (1.0, 0.0, 0.0), validate=True),  # wrong width
    ]
    raises("a bad record in a batch is rejected", InvalidVectorError,
           lambda: partial.add_many(bad_batch))
    check("a rejected batch leaves the store untouched", partial.count() == 1,
          str(partial.count()))

    raises("a non-record cannot be added", VectorStoreError,
           lambda: partial.add("not a record"))


# ===========================================================================
# 3. Rejection: duplicates, dimensions, bad values
# ===========================================================================
def test_rejection() -> None:
    print("\nRejection")

    store, chunks = populated_store()
    first = chunks[0]

    # -- duplicates --------------------------------------------------------
    raises("a duplicate chunk id is rejected", DuplicateChunkError,
           lambda: store.add(make_record(first, BASIS[first.section])))
    check("a rejected duplicate does not change the count", store.count() == 6)

    original = store.get(first.chunk_id)
    assert original is not None
    raises("a duplicate id with a different vector is still rejected",
           DuplicateChunkError,
           lambda: store.add(make_record(first, unit(0, 0, 1, 1))))
    check("duplicate insertion never silently overwrites",
          store.get(first.chunk_id).vector == original.vector)

    dupe = fixture_chunks()
    raises("a duplicate inside one batch is rejected", DuplicateChunkError,
           lambda: VectorStore().add_many([
               make_record(dupe[0], BASIS[dupe[0].section]),
               make_record(dupe[0], BASIS[dupe[0].section]),
           ]))

    # -- dimensions --------------------------------------------------------
    other = fixture_chunks("second-document", "Second Document")
    raises("a wrong-dimension vector is rejected", InvalidVectorError,
           lambda: store.add(make_record(other[0], unit(1, 0, 0))))
    raises("different embedding dimensions cannot be mixed", InvalidVectorError,
           lambda: store.add(make_record(other[1], unit(1, 0, 0, 0, 0))))
    raises("vectors from a different model cannot be mixed", InvalidVectorError,
           lambda: store.add(make_record(other[2], BASIS["Summary"],
                                         model_name="other/model")))

    # -- bad values --------------------------------------------------------
    raises("a zero vector is rejected", InvalidVectorError,
           lambda: store.add(make_record(other[0], (0.0, 0.0, 0.0, 0.0),
                                         normalized=False)))

    # These three cannot occur through normal construction -- EmbeddingResult
    # rejects them first -- so they are built with validation bypassed, which
    # is what proves the store checks for itself.
    raises("a NaN vector is rejected by the store itself", InvalidVectorError,
           lambda: store.add(make_record(other[0], (float("nan"), 0.0, 0.0, 1.0),
                                         normalized=False, validate=False)))
    raises("an infinite vector is rejected by the store itself", InvalidVectorError,
           lambda: store.add(make_record(other[1], (float("inf"), 0.0, 0.0, 1.0),
                                         normalized=False, validate=False)))
    raises("an empty vector is rejected by the store itself", InvalidVectorError,
           lambda: store.add(make_record(other[2], (), normalized=False,
                                         validate=False)))
    check("no rejected record was ever stored", store.count() == 6)

    # -- record-level pairing ---------------------------------------------
    mismatched = make_record(other[0], BASIS["Summary"])
    raises("an embedding paired with the wrong chunk is rejected", ValueError,
           lambda: VectorRecord(chunk=first, embedding=mismatched.embedding))
    raises("an unexpected field on a record is rejected", ValueError,
           lambda: VectorRecord(chunk=first, embedding=mismatched.embedding,
                                extra="value"))


# ===========================================================================
# 4. Cosine similarity, by hand
# ===========================================================================
def test_cosine() -> None:
    print("\nCosine similarity")

    check("identical vectors score 1",
          abs(cosine_similarity((1, 2, 3), (1, 2, 3)) - 1.0) < COSINE_TOLERANCE)
    check("opposite vectors score -1",
          abs(cosine_similarity((1, 0), (-1, 0)) + 1.0) < COSINE_TOLERANCE)
    check("orthogonal vectors score 0",
          abs(cosine_similarity((1, 0), (0, 1))) < COSINE_TOLERANCE)
    check("a 45-degree angle scores 1/sqrt(2)",
          abs(cosine_similarity((1, 1), (1, 0)) - (1 / math.sqrt(2))) < COSINE_TOLERANCE)
    check("cosine ignores magnitude",
          abs(cosine_similarity((1, 1), (5, 5)) - 1.0) < COSINE_TOLERANCE)
    check("cosine is symmetric",
          abs(cosine_similarity((3, 1), (1, 4)) - cosine_similarity((1, 4), (3, 1)))
          < COSINE_TOLERANCE)

    # The identity the embedding layer's normalisation buys us.
    a, b = unit(1, 2, 3, 4), unit(4, 3, 2, 1)
    dot = sum(x * y for x, y in zip(a, b))
    check("for normalised vectors, cosine equals the dot product",
          abs(cosine_similarity(a, b) - dot) < COSINE_TOLERANCE,
          f"{cosine_similarity(a, b)} vs {dot}")

    # And is still correct when they are not normalised.
    p, q = (3.0, 1.0, 0.0, 0.0), (1.0, 4.0, 0.0, 0.0)
    expected = (3 * 1 + 1 * 4) / (math.sqrt(10) * math.sqrt(17))
    check("for non-normalised vectors, cosine divides by both norms",
          abs(cosine_similarity(p, q) - expected) < COSINE_TOLERANCE,
          f"{cosine_similarity(p, q)} vs {expected}")
    check("a scaled vector gives the same score as its unit form",
          abs(cosine_similarity(p, q)
              - cosine_similarity(unit(*p), unit(*q))) < COSINE_TOLERANCE)

    raises("cosine against a zero vector is rejected", InvalidVectorError,
           lambda: cosine_similarity((0, 0, 0), (1, 2, 3)))
    raises("cosine with mismatched widths is rejected", InvalidVectorError,
           lambda: cosine_similarity((1, 2), (1, 2, 3)))
    raises("cosine with an empty vector is rejected", InvalidVectorError,
           lambda: cosine_similarity((), (1, 2, 3)))
    raises("cosine with NaN is rejected", InvalidVectorError,
           lambda: cosine_similarity((float("nan"), 1), (1, 2)))
    raises("cosine with infinity is rejected", InvalidVectorError,
           lambda: cosine_similarity((float("inf"), 1), (1, 2)))
    raises("cosine on non-numeric input is rejected", InvalidVectorError,
           lambda: cosine_similarity(("a", "b"), (1, 2)))


# ===========================================================================
# 5. Search
# ===========================================================================
def test_search() -> None:
    print("\nSearch")

    store, chunks = populated_store()
    by_section = {c.section: c for c in chunks}

    results = store.search(BASIS["Summary"], top_k=6)
    check("search returns results", len(results) == 6)
    check("every result is a SearchResult",
          all(isinstance(r, SearchResult) for r in results))
    check("the exact match ranks first",
          results[0].chunk_id == by_section["Summary"].chunk_id, results[0].chunk_id)
    check("the exact match scores 1.0", abs(results[0].similarity - 1.0) < 1e-12,
          str(results[0].similarity))
    check("results are sorted by descending similarity",
          all(a.similarity >= b.similarity for a, b in zip(results, results[1:])),
          str([round(r.similarity, 4) for r in results]))
    check("ranks are consecutive from zero",
          [r.rank for r in results] == list(range(len(results))))

    # Scores worked out by hand: Summary=(1,0,0,0) against the basis above.
    scores = {r.section: round(r.similarity, 6) for r in results}
    check("a 45-degree neighbour scores 0.707107",
          scores["What the DPI engine can observe"] == round(1 / math.sqrt(2), 6),
          str(scores))
    check("an orthogonal vector scores 0.0", scores["Indicators"] == 0.0, str(scores))
    check("the all-ones vector scores 0.5", scores["References"] == 0.5, str(scores))
    check("every score is within [-1, 1]",
          all(-1.0 <= r.similarity <= 1.0 for r in results))

    # -- top_k ------------------------------------------------------------
    check("top_k limits the result count", len(store.search(BASIS["Summary"], 3)) == 3)
    check("top_k=1 returns only the best match",
          len(store.search(BASIS["Summary"], 1)) == 1)
    check("top_k larger than the store returns everything",
          len(store.search(BASIS["Summary"], 999)) == store.count())
    check("top_k=0 explicitly returns nothing",
          store.search(BASIS["Summary"], 0) == ())
    raises("a negative top_k is rejected", ValueError,
           lambda: store.search(BASIS["Summary"], -1))
    raises("a non-integer top_k is rejected", ValueError,
           lambda: store.search(BASIS["Summary"], 2.5))
    raises("a boolean top_k is rejected", ValueError,
           lambda: store.search(BASIS["Summary"], True))

    # -- query validation -------------------------------------------------
    raises("a wrong-dimension query is rejected", InvalidVectorError,
           lambda: store.search((1.0, 0.0, 0.0), top_k=1))
    raises("a NaN query is rejected", InvalidVectorError,
           lambda: store.search((float("nan"), 0.0, 0.0, 1.0), top_k=1))
    raises("an infinite query is rejected", InvalidVectorError,
           lambda: store.search((float("inf"), 0.0, 0.0, 1.0), top_k=1))
    raises("a zero query is rejected", InvalidVectorError,
           lambda: store.search((0.0, 0.0, 0.0, 0.0), top_k=1))
    raises("an empty query is rejected", InvalidVectorError,
           lambda: store.search((), top_k=1))

    # -- threshold --------------------------------------------------------
    filtered = store.search(BASIS["Summary"], top_k=6, min_similarity=0.5)
    check("a minimum similarity threshold filters results", len(filtered) == 3,
          str([round(r.similarity, 4) for r in filtered]))
    check("the threshold is inclusive",
          any(abs(r.similarity - 0.5) < 1e-12 for r in filtered))
    check("ranks are renumbered after filtering",
          [r.rank for r in filtered] == [0, 1, 2])
    check("a threshold of 1.0 keeps only exact matches",
          len(store.search(BASIS["Summary"], 6, min_similarity=1.0)) == 1)
    # unit(1,1,1,0) matches nothing exactly: its best score is References at
    # 3 / (sqrt(3) * 2) = 0.866, so a 0.95 threshold must return nothing.
    check("a threshold above every score returns nothing",
          store.search(unit(1, 1, 1, 0), 6, min_similarity=0.95) == (),
          str([round(r.similarity, 4)
               for r in store.search(unit(1, 1, 1, 0), 6)]))
    raises("an out-of-range threshold is rejected", ValueError,
           lambda: store.search(BASIS["Summary"], 6, min_similarity=1.5))

    # -- provenance -------------------------------------------------------
    top = results[0]
    source = by_section["Summary"]
    check("results preserve the chunk id", top.chunk_id == source.chunk_id)
    check("results preserve the document id", top.document_id == source.document_id)
    check("results preserve the section", top.section == source.section)
    check("results preserve the title", top.record.title == source.title)
    check("results preserve the category", top.record.category is source.category)
    check("results preserve the text", top.text == source.text)
    check("results preserve the licence", top.record.licence == source.licence)
    check("results preserve the source list", top.record.sources == list(source.sources))
    check("results can cite their origin",
          source.document_id in top.citation() and source.section in top.citation())
    check("results carry the model name", top.record.model_name == MODEL_NAME)
    check("results carry the normalisation state", top.record.normalized is True)


# ===========================================================================
# 6. Ties, determinism and non-normalised vectors
# ===========================================================================
def test_ties_and_determinism() -> None:
    print("\nTies, determinism and normalisation")

    # Three documents whose chunks share one vector: every score ties exactly.
    store = VectorStore("ties")
    tied_chunks = []
    for doc_id, title in (("charlie-document", "Charlie"), ("alpha-document", "Alpha"),
                          ("bravo-document", "Bravo")):
        chunk = fixture_chunks(doc_id, title)[0]
        tied_chunks.append(chunk)
        store.add(make_record(chunk, unit(1, 1, 0, 0)))

    results = store.search(unit(1, 1, 0, 0), top_k=3)
    check("tied scores are genuinely equal",
          len({round(r.similarity, 12) for r in results}) == 1,
          str([r.similarity for r in results]))
    check("ties break on chunk_id ascending",
          [r.chunk_id for r in results] == sorted(r.chunk_id for r in results),
          str([r.chunk_id for r in results]))
    # Inserted charlie, alpha, bravo -- so id order and insertion order differ,
    # and the result must follow the id.
    check("tie ordering ignores insertion order",
          [r.chunk_id for r in results] != [c.chunk_id for c in tied_chunks],
          str([r.chunk_id.split("#")[0] for r in results]))
    check("repeated searches return identical ordering",
          [r.chunk_id for r in store.search(unit(1, 1, 0, 0), 3)]
          == [r.chunk_id for r in results])

    # -- repeatability on the main fixture --------------------------------
    main, _ = populated_store()
    first = main.search(BASIS["References"], top_k=6)
    second = main.search(BASIS["References"], top_k=6)
    check("repeated search produces identical results",
          [(r.chunk_id, r.similarity) for r in first]
          == [(r.chunk_id, r.similarity) for r in second])

    # -- normalised vs non-normalised -------------------------------------
    chunks = fixture_chunks("scaled-document", "Scaled")
    scaled = VectorStore("scaled")
    scaled.add_many([
        make_record(chunk, tuple(c * (index + 2) for c in BASIS[chunk.section]),
                    normalized=False)
        for index, chunk in enumerate(chunks)
    ])
    check("a store of non-normalised vectors is accepted", scaled.count() == 6)
    check("non-normalised vectors are recorded as such",
          all(not r.normalized for r in scaled.records()))

    normal, normal_chunks = populated_store("normal")
    scaled_order = [r.section for r in scaled.search(BASIS["Summary"], 6)]
    normal_order = [r.section for r in normal.search(BASIS["Summary"], 6)]
    check("scaling the stored vectors does not change the ranking",
          scaled_order == normal_order, f"{scaled_order} vs {normal_order}")

    scaled_scores = [round(r.similarity, 9) for r in scaled.search(BASIS["Summary"], 6)]
    normal_scores = [round(r.similarity, 9) for r in normal.search(BASIS["Summary"], 6)]
    check("scaling the stored vectors does not change the scores",
          scaled_scores == normal_scores, f"{scaled_scores} vs {normal_scores}")

    # For normalised vectors the cosine score is the plain dot product.
    query = BASIS["References"]
    for result in normal.search(query, top_k=6):
        dot = float(np.dot(np.array(result.record.vector), np.array(query)))
        check(f"{result.section}: cosine equals the dot product for unit vectors",
              abs(result.similarity - dot) < COSINE_TOLERANCE,
              f"{result.similarity} vs {dot}")

    # A non-normalised query must also work.
    long_query = tuple(c * 7.5 for c in BASIS["Summary"])
    check("a non-normalised query gives the same ranking",
          [r.chunk_id for r in normal.search(long_query, 6)]
          == [r.chunk_id for r in normal.search(BASIS["Summary"], 6)])


# ===========================================================================
# 7. Matrix, mutation protection, clear, export
# ===========================================================================
def test_matrix_and_lifecycle() -> None:
    print("\nMatrix, mutation protection and lifecycle")

    store, chunks = populated_store()

    matrix = store.matrix()
    check("the matrix is N x D", matrix.shape == (6, DIMENSION), str(matrix.shape))
    check("matrix rows are the stored vectors in insertion order",
          all(np.allclose(matrix[index], np.array(BASIS[chunk.section]))
              for index, chunk in enumerate(chunks)))

    # -- protection -------------------------------------------------------
    matrix[0][0] = 999.0
    check("matrix() hands out a copy, not the live index",
          store.matrix()[0][0] != 999.0)
    check("mutating the copy leaves search results unchanged",
          abs(store.search(BASIS["Summary"], 1)[0].similarity - 1.0) < 1e-12)

    def write_through() -> None:
        store._build_matrix()[0][0] = 42.0  # noqa: SLF001 - deliberate probe

    raises("the internal matrix is read-only", ValueError, write_through)

    def write_row() -> None:
        store._vectors[0][0] = 42.0  # noqa: SLF001 - deliberate probe

    raises("stored vectors are read-only", ValueError, write_row)

    # -- search does not mutate -------------------------------------------
    before = store.matrix().tobytes()
    ids_before = store.chunk_ids()
    store.search(BASIS["Indicators"], top_k=6)
    store.search(BASIS["References"], top_k=2, min_similarity=0.1)
    check("search does not mutate the stored vectors",
          store.matrix().tobytes() == before)
    check("search does not change the record set", store.chunk_ids() == ids_before)
    check("search does not change the count", store.count() == 6)

    # -- clear ------------------------------------------------------------
    store.clear()
    check("clear() empties the store", store.count() == 0)
    check("clear() releases the dimension commitment", store.dimension is None)
    check("clear() releases the model commitment", store.model_name is None)
    check("clear() empties the id index", store.chunk_ids() == ())
    check("searching a cleared store returns nothing",
          store.search(BASIS["Summary"], 5) == ())

    reused = fixture_chunks("reused-document", "Reused")
    store.add(make_record(reused[0], unit(1, 0, 0, 0, 0, 0), model_name="other/model"))
    check("a cleared store accepts a different dimension and model",
          store.dimension == 6 and store.model_name == "other/model")

    # -- export -----------------------------------------------------------
    exported, _ = populated_store("export")
    dump = exported.to_json()
    parsed = json.loads(dump)
    check("export reports the record count", parsed["count"] == 6)
    check("export reports the dimension and model",
          parsed["dimension"] == DIMENSION and parsed["model_name"] == MODEL_NAME)
    check("export omits vectors by default",
          all("vector" not in entry for entry in parsed["records"]))
    check("export preserves provenance",
          all({"chunk_id", "document_id", "section", "licence"} <= set(entry)
              for entry in parsed["records"]))
    check("export is sorted by chunk id",
          [e["chunk_id"] for e in parsed["records"]]
          == sorted(e["chunk_id"] for e in parsed["records"]))
    check("export is deterministic", dump == exported.to_json())

    reordered = VectorStore("export")
    reordered.add_many([make_record(c, BASIS[c.section])
                        for c in reversed(fixture_chunks())])
    check("export is independent of insertion order", reordered.to_json() == dump)

    with_vectors = json.loads(exported.to_json(include_vectors=True))
    check("vectors can be included on request",
          all(len(entry["vector"]) == DIMENSION for entry in with_vectors["records"]))

    with tempfile.TemporaryDirectory() as raw:
        path = exported.export_json(Path(raw) / "index.json")
        check("export_json writes a file", path.is_file())
        check("the written file matches to_json()",
              path.read_text(encoding="utf-8") == dump)


# ===========================================================================
# 8. Isolation
# ===========================================================================
def test_isolation() -> None:
    print("\nIsolation")

    forbidden = [name for name in (
        "faiss", "chromadb", "qdrant_client", "pinecone", "weaviate", "sqlite3",
        "langchain", "langchain_community", "llama_index",
    ) if name in sys.modules]
    check("no vector database library is imported", not forbidden, f"imported: {forbidden}")
    check("no LangChain library is imported",
          not any(name.startswith("langchain") for name in sys.modules))
    check("the openai SDK is not imported", "openai" not in sys.modules)

    source = Path("ai/rag/vector_store.py").read_text(encoding="utf-8")
    for banned in ("import openai", "groq", "faiss", "chromadb", "langchain",
                   "sqlite", "requests", "urllib"):
        check(f"the vector store never references {banned!r}", banned not in source)
    check("the vector store does not import the embedding model class",
          "EmbeddingModel" not in source)
    check("the vector store does not load documents itself",
          "load_corpus" not in source)

    # Prove the network is untouched rather than asserting it.
    real_socket, real_connect = socket.socket, socket.create_connection

    def refuse(*args, **kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("the vector store attempted a network connection")

    socket.socket, socket.create_connection = refuse, refuse  # type: ignore[assignment]
    try:
        store, _ = populated_store("offline")
        results = store.search(BASIS["Summary"], top_k=3)
        check("indexing and searching make no network call", len(results) == 3)
    finally:
        socket.socket, socket.create_connection = real_socket, real_connect

    check("a synthetic corpus indexes and searches with no embedding model",
          populated_store("synthetic")[0].count() == 6)


# ===========================================================================
# 9. Integration -- the real corpus, embedded once
# ===========================================================================
def test_integration() -> None:
    print(f"\nIntegration -- real corpus with {DEFAULT_MODEL} (optional)")

    labels = (
        "the real corpus loads as six documents",
        "the real corpus chunks into 37 chunks",
        "every chunk is indexed",
        "the index holds the model's dimension",
        "the index matrix is 37 x 384",
        "a chunk's own embedding retrieves itself first",
        "self-similarity is 1.0",
        "results cite a real document and section",
        "real normalised vectors satisfy cosine == dot",
        "top_k is respected on the real index",
        "repeated search on the real index is identical",
    )

    if not sentence_transformers_available():
        for label in labels:
            skip(label, "sentence-transformers is not installed "
                        "(pip install -r requirements-rag.txt)")
        return

    model = EmbeddingModel(EmbeddingConfig())
    try:
        model.load()
    except ModelUnavailableError as exc:
        for label in labels:
            skip(label, f"{DEFAULT_MODEL} could not be loaded: {str(exc)[:100]}")
        return

    corpus = load_corpus()
    chunks = chunk_corpus(corpus)

    # The corpus is embedded exactly once for the whole integration tier.
    embeddings = model.embed_chunks(chunks)

    store = VectorStore("knowledge")
    store.add_many([VectorRecord(chunk=chunk, embedding=embedding)
                    for chunk, embedding in zip(chunks, embeddings)])

    check(labels[0], len(corpus) == 6, str(len(corpus)))
    check(labels[1], len(chunks) == 37, str(len(chunks)))
    check(labels[2], store.count() == len(chunks) == 37, str(store.count()))
    check(labels[3], store.dimension == DEFAULT_DIMENSION, str(store.dimension))
    check(labels[4], store.matrix().shape == (37, DEFAULT_DIMENSION),
          str(store.matrix().shape))

    # Self-retrieval: use a stored vector as the query.  No query construction
    # is involved -- that belongs to a later step.
    target = next(e for e in embeddings if e.document_id == "dns-tunneling"
                  and e.section == "Indicators")
    hits = store.search(target.vector, top_k=5)
    check(labels[5], hits[0].chunk_id == target.chunk_id, hits[0].chunk_id)
    check(labels[6], abs(hits[0].similarity - 1.0) < 1e-9, str(hits[0].similarity))
    check(labels[7],
          hits[0].document_id == "dns-tunneling" and hits[0].section == "Indicators")

    dot = float(np.dot(np.array(hits[1].record.vector), np.array(target.vector)))
    check(labels[8], abs(hits[1].similarity - dot) < 1e-6,
          f"{hits[1].similarity} vs {dot}")

    check(labels[9], len(store.search(target.vector, top_k=3)) == 3)
    check(labels[10],
          [r.chunk_id for r in store.search(target.vector, 5)]
          == [r.chunk_id for r in hits])

    print(f"        indexed {store.count()} chunks from {len(corpus)} documents, "
          f"dim={store.dimension}, top match {hits[0].similarity:.4f} "
          f"({hits[0].citation()})")
    print(f"        next nearest: {hits[1].similarity:.4f} ({hits[1].citation()})")


# ===========================================================================
def main() -> int:
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print(f"RAG step 4 -- local vector store (numpy {np.__version__})")

    test_empty_store()
    test_adding()
    test_rejection()
    test_cosine()
    test_search()
    test_ties_and_determinism()
    test_matrix_and_lifecycle()
    test_isolation()
    test_integration()

    total = _passed + _failed
    suffix = f", {_skipped} skipped" if _skipped else ""
    print(f"\n{_passed}/{total} checks passed{suffix}")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
