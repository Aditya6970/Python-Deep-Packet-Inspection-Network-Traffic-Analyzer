"""Test runner for the embedding layer -- RAG step 3 only.

Two tiers, deliberately separated
---------------------------------
**Unit tier** -- always runs.  Configuration, the embedding-input builder,
batching, ordering, normalisation bookkeeping, provenance, error paths and
serialisation.  Needs nothing installed beyond pydantic: the batching and
ordering logic is exercised through the ``encoder`` seam using a deterministic
stub defined *in this file*.  That stub is a test double, never reachable from
the library -- :mod:`ai.rag.embeddings` has no fallback encoder and constructs
one only from a real model.

**Integration tier** -- runs only when ``sentence-transformers`` is installed
*and* ``BAAI/bge-small-en-v1.5`` can actually be loaded.  Otherwise every
integration check SKIPs with the reason, and the suite still passes.  A missing
model download is an environment fact, not a defect.

Resource use is deliberate: the real 37-chunk corpus is embedded **once** and
the results are reused across checks.  The equivalence and stability checks run
on three synthetic chunks rather than re-embedding the corpus.

Numeric tolerance
-----------------
Embeddings of identical text are compared component-wise within
``NUMERIC_TOLERANCE`` (1e-5), never for exact equality: float32 reductions are
not bit-reproducible across BLAS builds, thread counts or devices.

Run::

    python run_rag_embedding_tests.py
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

# Import the package so its console-encoding fix is applied before any output.
import dpi  # noqa: F401
from ai.rag.chunking import ChunkConfig, KnowledgeChunk, chunk_corpus, chunk_document
from ai.rag.documents import CATEGORY_DIRECTORIES, REQUIRED_SECTIONS, Category, load_corpus
from ai.rag.embeddings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    ENV_BATCH_SIZE,
    ENV_MODEL,
    NUMERIC_TOLERANCE,
    EmbeddingConfig,
    EmbeddingInputError,
    EmbeddingModel,
    EmbeddingResult,
    ModelUnavailableError,
    build_embedding_input,
    embedding_statistics,
    l2_norm,
    sentence_transformers_available,
    serialize_embedding_metadata,
)

_passed = 0
_failed = 0
_skipped = 0


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


def close_enough(a, b, tolerance: float = NUMERIC_TOLERANCE) -> bool:
    """Component-wise comparison of two vectors within ``tolerance``."""
    return len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b))


# ===========================================================================
# Test double
# ===========================================================================
class StubEncoder:
    """A deterministic stand-in for a real model, used by the unit tier only.

    Vectors are derived from a SHA-256 of the input text, so they are stable
    and text-sensitive -- enough to prove that batching preserves order and
    that each input maps to its own vector.  They carry no semantic meaning
    whatsoever, which is exactly why nothing in ``ai/rag/embeddings.py`` can
    ever select this: a hashed vector would build an index that looks healthy
    and retrieves nonsense.
    """

    def __init__(self, dimension: int = 8, name: str = "stub/deterministic") -> None:
        self._dimension = dimension
        self._name = name
        self.calls = 0
        self.batch_sizes: list[int] = []

    @property
    def name(self) -> str:
        return self._name

    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts, normalize: bool):
        self.calls += 1
        self.batch_sizes.append(len(texts))
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            row = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0
                   for i in range(self._dimension)]
            if normalize:
                length = l2_norm(row) or 1.0
                row = [component / length for component in row]
            rows.append(row)
        return rows


# ===========================================================================
# Fixtures
# ===========================================================================
FRONT_MATTER = """\
id: {doc_id}
title: {title}
category: protocols
version: 1.0
updated: 2026-08-27
applies_to:
  - dns_high_volume
keywords:
  - sample
mitre: []
severity_hint: info
sources:
  - Authored for this project.
licence: project-authored
"""


def build_document_text(doc_id: str = "synthetic-document",
                        title: str = "Synthetic Document") -> str:
    sections = "\n".join(
        f"## {name}\n\nBody text for {name} in the synthetic fixture.\n"
        for name in REQUIRED_SECTIONS
    )
    return f"---\n{FRONT_MATTER.format(doc_id=doc_id, title=title)}---\n\n{sections}"


def synthetic_chunks() -> tuple[KnowledgeChunk, ...]:
    """Chunks built from a temporary document, independent of the real corpus."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        for name in CATEGORY_DIRECTORIES:
            (root / name).mkdir(exist_ok=True)
        (root / "protocols" / "synthetic-document.md").write_text(
            build_document_text(), encoding="utf-8"
        )
        return chunk_document(load_corpus(root)[0], ChunkConfig())


# ===========================================================================
# 1. Configuration
# ===========================================================================
def test_configuration() -> None:
    print("\nConfiguration")

    config = EmbeddingConfig()
    check("the default model is bge-small-en-v1.5",
          config.model_name == DEFAULT_MODEL == "BAAI/bge-small-en-v1.5")
    check("the default batch size is documented and configurable",
          config.batch_size == DEFAULT_BATCH_SIZE == 16)
    check("normalisation is on by default", config.normalize is True)
    check("offline loading is off by default", config.local_files_only is False)
    check("the config is immutable",
          type(config).__dataclass_params__.frozen)  # type: ignore[attr-defined]

    check("the model is configurable, not hard-coded",
          EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2").model_name
          == "sentence-transformers/all-MiniLM-L6-v2")
    check("a local directory is accepted as a model name",
          EmbeddingConfig(model_name="./models/bge-small").model_name == "./models/bge-small")

    raises("an empty model name is rejected", ValueError,
           lambda: EmbeddingConfig(model_name=""))
    raises("a whitespace model name is rejected", ValueError,
           lambda: EmbeddingConfig(model_name="   "))
    raises("a model name with spaces is rejected", ValueError,
           lambda: EmbeddingConfig(model_name="not a model name"))
    raises("a zero batch size is rejected", ValueError,
           lambda: EmbeddingConfig(batch_size=0))
    raises("a negative batch size is rejected", ValueError,
           lambda: EmbeddingConfig(batch_size=-4))
    raises("a non-integer batch size is rejected", ValueError,
           lambda: EmbeddingConfig(batch_size="16"))
    raises("an empty device string is rejected", ValueError,
           lambda: EmbeddingConfig(device="  "))

    # -- environment ------------------------------------------------------
    previous = {name: os.environ.get(name) for name in (ENV_MODEL, ENV_BATCH_SIZE)}
    try:
        os.environ[ENV_MODEL] = "BAAI/bge-base-en-v1.5"
        os.environ[ENV_BATCH_SIZE] = "4"
        from_env = EmbeddingConfig.from_env()
        check("the model name can be set from the environment",
              from_env.model_name == "BAAI/bge-base-en-v1.5")
        check("the batch size can be set from the environment", from_env.batch_size == 4)
        check("explicit arguments beat the environment",
              EmbeddingConfig.from_env(model_name=DEFAULT_MODEL).model_name == DEFAULT_MODEL)

        os.environ[ENV_BATCH_SIZE] = "not-a-number"
        raises("a malformed batch size in the environment is rejected", ValueError,
               EmbeddingConfig.from_env)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    check("a clean environment yields the documented defaults",
          EmbeddingConfig.from_env().model_name == DEFAULT_MODEL)


# ===========================================================================
# 2. The embedding input string
# ===========================================================================
def test_embedding_input() -> None:
    print("\nEmbedding input")

    chunks = chunk_corpus()
    sample = next(c for c in chunks if c.document_id == "dns-tunneling"
                  and c.section == "Indicators")
    text = build_embedding_input(sample)

    check("the input is deterministic", text == build_embedding_input(sample))
    check("the input is deterministic across a rebuild of the chunk",
          text == build_embedding_input(
              next(c for c in chunk_corpus() if c.chunk_id == sample.chunk_id)))

    check("the input contains the document title", f"Title: {sample.title}" in text)
    check("the input contains the category", f"Category: {sample.category.value}" in text)
    check("the input contains the section", f"Section: {sample.section}" in text)
    check("the input contains the chunk body", sample.text in text)
    check("provenance comes before the body",
          text.index("Section:") < text.index(sample.text[:40]))

    # Metadata that would distort similarity is deliberately left out.
    check("the licence is not embedded", sample.licence not in text)
    check("the source list is not embedded",
          all(source not in text for source in sample.sources))
    check("keywords are not embedded as a block",
          ", ".join(sample.keywords) not in text)

    # -- no secrets, no runtime data --------------------------------------
    os.environ["DPI_EMBED_TEST_SECRET"] = "sk-should-never-appear-anywhere"
    try:
        inputs = [build_embedding_input(c) for c in chunks]
        joined = "\n".join(inputs)
        check("no environment value leaks into the input",
              "sk-should-never-appear-anywhere" not in joined)
    finally:
        os.environ.pop("DPI_EMBED_TEST_SECRET", None)

    for marker in ("sk-", "api_key", "API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY",
                   "Bearer ", "password"):
        check(f"no {marker!r} appears in any embedding input", marker not in joined)

    check("no absolute filesystem path leaks into the input",
          "/home/" not in joined and "C:\\" not in joined)
    check("every corpus chunk produces a non-empty input",
          all(t.strip() for t in inputs))
    check("inputs are unique per chunk", len(set(inputs)) == len(inputs))

    raises("a non-chunk object is rejected", EmbeddingInputError,
           lambda: build_embedding_input("just a string"))
    raises("None is rejected", EmbeddingInputError, lambda: build_embedding_input(None))
    raises("a dict masquerading as a chunk is rejected", EmbeddingInputError,
           lambda: build_embedding_input({"text": "hello"}))


# ===========================================================================
# 3. Batching, ordering and bookkeeping (stub encoder)
# ===========================================================================
def test_batching_with_stub() -> None:
    print("\nBatching and ordering (stub encoder)")

    chunks = synthetic_chunks()
    check("the synthetic fixture produced chunks", len(chunks) == len(REQUIRED_SECTIONS))

    stub = StubEncoder()
    model = EmbeddingModel(EmbeddingConfig(batch_size=2), encoder=stub)
    results = model.embed_chunks(chunks)

    check("one embedding per chunk", len(results) == len(chunks))
    check("output order matches input order",
          [r.chunk_id for r in results] == [c.chunk_id for c in chunks])
    check("batches were actually used", stub.calls == 3, f"{stub.calls} encode calls")
    check("batch sizes respect the configured maximum",
          all(size <= 2 for size in stub.batch_sizes), str(stub.batch_sizes))
    check("batches cover every input exactly once", sum(stub.batch_sizes) == len(chunks))

    # Batch size must be a memory strategy, not a reordering.
    for size in (1, 3, 6, 64):
        other = EmbeddingModel(EmbeddingConfig(batch_size=size), encoder=StubEncoder())
        produced = other.embed_chunks(chunks)
        check(f"batch size {size} preserves order",
              [r.chunk_id for r in produced] == [c.chunk_id for c in chunks])
        check(f"batch size {size} produces identical vectors",
              all(close_enough(a.vector, b.vector) for a, b in zip(produced, results)))

    # Single-item and batch paths must agree.
    singles = []
    for chunk in chunks:
        one = EmbeddingModel(EmbeddingConfig(batch_size=1), encoder=StubEncoder())
        singles.extend(one.embed_chunks([chunk]))
    check("single-item embedding equals batch embedding",
          all(close_enough(s.vector, b.vector) for s, b in zip(singles, results)))

    # Repeat runs must be stable.
    repeat = EmbeddingModel(EmbeddingConfig(batch_size=2), encoder=StubEncoder())
    again = repeat.embed_chunks(chunks)
    check("repeated embedding is stable within tolerance",
          all(close_enough(a.vector, b.vector) for a, b in zip(again, results)))

    check("dimensions are consistent across all results",
          len({r.dimension for r in results}) == 1)
    check("the vector length matches the reported dimension",
          all(len(r.vector) == r.dimension for r in results))
    check("the configured model name is recorded on every result",
          all(r.model_name == DEFAULT_MODEL for r in results))


# ===========================================================================
# 4. Normalisation bookkeeping
# ===========================================================================
def test_normalisation_with_stub() -> None:
    print("\nNormalisation (stub encoder)")

    chunks = synthetic_chunks()[:3]

    normalised = EmbeddingModel(EmbeddingConfig(normalize=True),
                                encoder=StubEncoder()).embed_chunks(chunks)
    check("normalised vectors have unit L2 norm",
          all(abs(l2_norm(r.vector) - 1.0) <= 1e-6 for r in normalised),
          str([round(l2_norm(r.vector), 8) for r in normalised]))
    check("normalisation is recorded on the result",
          all(r.normalized is True for r in normalised))

    raw = EmbeddingModel(EmbeddingConfig(normalize=False),
                         encoder=StubEncoder()).embed_chunks(chunks)
    check("un-normalised vectors are not silently normalised",
          any(abs(l2_norm(r.vector) - 1.0) > 1e-3 for r in raw),
          str([round(l2_norm(r.vector), 4) for r in raw]))
    check("the non-normalised state is recorded too",
          all(r.normalized is False for r in raw))

    # The model rejects a result whose flag disagrees with its vector.
    payload = normalised[0].model_dump()
    payload["vector"] = tuple(component * 3.0 for component in payload["vector"])
    raises("a vector marked normalised but not unit-length is rejected", ValueError,
           lambda: EmbeddingResult.model_validate(payload))


# ===========================================================================
# 5. Provenance and the result model
# ===========================================================================
def test_provenance_with_stub() -> None:
    print("\nProvenance and result model (stub encoder)")

    chunks = synthetic_chunks()
    results = EmbeddingModel(EmbeddingConfig(), encoder=StubEncoder()).embed_chunks(chunks)

    for chunk, result in zip(chunks, results):
        ok = (
            result.chunk_id == chunk.chunk_id
            and result.document_id == chunk.document_id
            and result.category is chunk.category
            and result.section == chunk.section
            and result.heading_path == chunk.heading_path
            and result.content_sha256 == chunk.content_sha256
        )
        check(f"{chunk.section}: provenance is carried onto the embedding", ok)

    check("the input hash matches the text actually embedded",
          all(r.input_sha256 == hashlib.sha256(
              build_embedding_input(c).encode("utf-8")).hexdigest()
              for c, r in zip(chunks, results)))
    check("chunk ids stay associated with their own vectors",
          all(r.chunk_id == c.chunk_id for c, r in zip(chunks, results)))

    # Association must survive reordering: look up by id, not by position.
    by_id = {r.chunk_id: r for r in results}
    shuffled = list(reversed(chunks))
    reshuffled = EmbeddingModel(EmbeddingConfig(),
                                encoder=StubEncoder()).embed_chunks(shuffled)
    check("a chunk embedded in a different position gets the same vector",
          all(close_enough(r.vector, by_id[r.chunk_id].vector) for r in reshuffled))

    # -- rejection cases --------------------------------------------------
    sample = results[0].model_dump()

    def variant(**changes):
        data = dict(sample)
        data.update(changes)
        return lambda: EmbeddingResult.model_validate(data)

    raises("an unexpected field on the result is rejected", ValueError,
           variant(unexpected_field="value"))
    raises("an empty vector is rejected", ValueError, variant(vector=()))
    raises("a dimension that disagrees with the vector is rejected", ValueError,
           variant(dimension=999))
    raises("a NaN component is rejected", ValueError,
           variant(vector=(float("nan"),) * sample["dimension"]))
    raises("an infinite component is rejected", ValueError,
           variant(vector=(float("inf"),) * sample["dimension"]))
    raises("a missing model name is rejected", ValueError, variant(model_name=""))

    def mutate() -> None:
        results[0].model_name = "other"  # type: ignore[misc]

    raises("an embedding result cannot be mutated", ValueError, mutate)

    check("metadata() omits the vector", "vector" not in results[0].metadata())
    check("metadata() keeps the provenance",
          results[0].metadata()["chunk_id"] == chunks[0].chunk_id)


# ===========================================================================
# 6. Empty input, load-once, serialisation
# ===========================================================================
def test_contract_with_stub() -> None:
    print("\nContract (stub encoder)")

    chunks = synthetic_chunks()
    stub = StubEncoder()
    model = EmbeddingModel(EmbeddingConfig(), encoder=stub)

    raises("an empty chunk list is rejected", EmbeddingInputError,
           lambda: model.embed_chunks([]))
    raises("an empty text list is rejected", EmbeddingInputError,
           lambda: model.embed_texts([]))
    raises("an empty string is rejected", EmbeddingInputError,
           lambda: model.embed_texts([""]))
    raises("a whitespace-only string is rejected", EmbeddingInputError,
           lambda: model.embed_texts(["   \n "]))
    raises("a non-string input is rejected", EmbeddingInputError,
           lambda: model.embed_texts([123]))
    raises("an invalid chunk object is rejected", EmbeddingInputError,
           lambda: model.embed_chunks(["not a chunk"]))

    # -- load once --------------------------------------------------------
    before = model.load_count
    model.embed_chunks(chunks)
    model.embed_chunks(chunks)
    model.embed_texts(["one more string"])
    check("the model is loaded once and reused, not per call",
          model.load_count == before, f"load_count went {before} -> {model.load_count}")
    check("the model reports itself as loaded", model.loaded is True)

    # -- serialisation ----------------------------------------------------
    results = model.embed_chunks(chunks)
    first = serialize_embedding_metadata(results)
    second = serialize_embedding_metadata(
        EmbeddingModel(EmbeddingConfig(), encoder=StubEncoder()).embed_chunks(chunks)
    )
    check("metadata serialisation is deterministic", first == second)
    check("serialised metadata is valid JSON", isinstance(json.loads(first), list))
    check("serialised metadata contains no vectors",
          all("vector" not in entry for entry in json.loads(first)))
    check("serialised metadata names the model and dimension",
          json.loads(first)[0]["model_name"] == DEFAULT_MODEL
          and json.loads(first)[0]["dimension"] == 8)

    stats = embedding_statistics(results)
    check("statistics report the embedding count", stats["embeddings"] == len(chunks))
    check("statistics report the normalisation state", stats["normalized"] is True)
    check("statistics on an empty set do not raise",
          embedding_statistics([])["embeddings"] == 0)


# ===========================================================================
# 7. Isolation: no providers, no keys, no vector databases
# ===========================================================================
def test_isolation() -> None:
    print("\nIsolation")

    forbidden = [name for name in (
        "faiss", "chromadb", "qdrant_client", "pinecone", "weaviate",
        "langchain", "langchain_community", "llama_index",
    ) if name in sys.modules]
    check("no vector database library is imported", not forbidden, f"imported: {forbidden}")
    check("no LangChain library is imported",
          not any(name.startswith("langchain") for name in sys.modules))
    check("the openai SDK is not imported by the embedding layer",
          "openai" not in sys.modules)

    source = Path("ai/rag/embeddings.py").read_text(encoding="utf-8")
    for banned in ("import openai", "from openai", "groq", "api_key", "API_KEY"):
        check(f"the embedding module never references {banned!r}",
              banned not in source.replace("no API key", "").replace(
                  "No API key", "").replace("without an API key", ""))

    # No API key present, and no provider variable consulted.
    saved = {name: os.environ.pop(name, None)
             for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "DPI_LLM_PROVIDER")}
    try:
        chunks = synthetic_chunks()[:2]
        results = EmbeddingModel(EmbeddingConfig(),
                                 encoder=StubEncoder()).embed_chunks(chunks)
        check("embedding works with no API key in the environment", len(results) == 2)

        # Prove it rather than assert it.
        real_socket, real_connect = socket.socket, socket.create_connection

        def refuse(*args, **kwargs):  # pragma: no cover - only runs on failure
            raise AssertionError("the embedding path attempted a network connection")

        socket.socket, socket.create_connection = refuse, refuse  # type: ignore[assignment]
        try:
            offline = EmbeddingModel(EmbeddingConfig(),
                                     encoder=StubEncoder()).embed_chunks(chunks)
            check("embedding an already-loaded model makes no network call",
                  len(offline) == 2)
        finally:
            socket.socket, socket.create_connection = real_socket, real_connect
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


# ===========================================================================
# 8. Loading errors -- no silent fallback
# ===========================================================================
def test_load_errors() -> None:
    print("\nModel loading errors")

    if not sentence_transformers_available():
        model = EmbeddingModel(EmbeddingConfig())
        raises("a missing sentence-transformers install is reported clearly",
               ModelUnavailableError, model.load)
        skip("a nonexistent model name is rejected", "sentence-transformers not installed")
        return

    bogus = EmbeddingModel(EmbeddingConfig(
        model_name="definitely-not-a-real-org/definitely-not-a-real-model"))
    raises("a nonexistent model name is reported clearly, with no fallback",
           ModelUnavailableError, bogus.load)
    check("a failed load leaves the model unloaded", bogus.loaded is False)
    check("a failed load does not count as a load", bogus.load_count == 0)

    offline = EmbeddingModel(EmbeddingConfig(
        model_name="definitely-not-a-real-org/definitely-not-a-real-model",
        local_files_only=True))
    try:
        offline.load()
        check("offline loading of an uncached model fails", False, "it loaded")
    except ModelUnavailableError as exc:
        check("offline loading of an uncached model explains the cache miss",
              "offline" in str(exc).lower() or "cache" in str(exc).lower(), str(exc)[:120])


# ===========================================================================
# 9. Adapter plumbing against a real SentenceTransformer, with no network
# ===========================================================================
def build_tiny_local_model(directory: Path):
    """Assemble a genuinely tiny SentenceTransformer from scratch, offline.

    A 16-dimension, one-layer BERT over a hand-written 90-token vocabulary,
    saved to ``directory`` and loaded back through the normal
    ``SentenceTransformer`` machinery.  Weights are untrained, so its vectors
    carry no meaning at all -- and nothing here asserts that they do.

    Its whole purpose is to exercise the *adapter*: that
    ``_SentenceTransformerEncoder`` calls ``encode`` with parameters this
    library version accepts, that shapes and ordering come back as expected,
    and that the normalisation flag reaches the model.  Without it, the first
    time that code path runs on a real sentence-transformer is on the user's
    machine, which is a poor place to discover a renamed keyword argument.
    """
    from transformers import BertConfig, BertModel, BertTokenizer

    vocab = (["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
             + [f"tok{i}" for i in range(60)] + list("abcdefghijklmnopqrstuvwxyz"))
    (directory / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")

    tokenizer = BertTokenizer(vocab_file=str(directory / "vocab.txt"))
    config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32,
                        max_position_embeddings=64)
    BertModel(config).save_pretrained(directory)
    tokenizer.save_pretrained(directory)


def test_adapter_plumbing() -> None:
    print("\nSentenceTransformer adapter (tiny local model, no network)")

    if not sentence_transformers_available():
        skip("the adapter drives a real SentenceTransformer",
             "sentence-transformers is not installed")
        return

    import contextlib
    import io

    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        noise = io.StringIO()
        try:
            with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
                build_tiny_local_model(directory)
        except Exception as exc:  # noqa: BLE001 - environment, not a defect
            skip("the adapter drives a real SentenceTransformer",
                 f"could not build a local model: {type(exc).__name__}: {exc}")
            return

        chunks = synthetic_chunks()[:4]
        config = EmbeddingConfig(model_name=str(directory), batch_size=2,
                                 local_files_only=True)
        model = EmbeddingModel(config)

        with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
            model.load()
            results = model.embed_chunks(chunks)

        check("a local directory loads as a model", model.loaded is True)
        check("offline loading works when the model is present locally",
              model.load_count == 1)
        check("the adapter reports the model's real dimension",
              model.dimension() == 16, str(model.dimension()))
        check("the adapter returns one vector per input", len(results) == len(chunks))
        check("the adapter preserves order",
              [r.chunk_id for r in results] == [c.chunk_id for c in chunks])
        check("the adapter returns vectors of the reported width",
              all(len(r.vector) == 16 for r in results))
        check("normalize_embeddings reaches the real model",
              all(abs(l2_norm(r.vector) - 1.0) <= 1e-4 for r in results),
              str([round(l2_norm(r.vector), 6) for r in results]))
        check("the local model name is recorded", results[0].model_name == str(directory))

        with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
            unnormalised = EmbeddingModel(
                EmbeddingConfig(model_name=str(directory), normalize=False,
                                local_files_only=True)
            )
            unnormalised._encoder = model._encoder  # reuse; do not rebuild
            raw_results = unnormalised.embed_chunks(chunks[:2])
        check("turning normalisation off reaches the real model too",
              any(abs(l2_norm(r.vector) - 1.0) > 1e-3 for r in raw_results),
              str([round(l2_norm(r.vector), 4) for r in raw_results]))

        with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
            repeat = model.embed_chunks(chunks)
        check("a real model re-embeds identical text within tolerance",
              all(close_enough(a.vector, b.vector) for a, b in zip(repeat, results)),
              f"tolerance {NUMERIC_TOLERANCE}")
        check("the real model was loaded exactly once", model.load_count == 1)


# ===========================================================================
# 10. Integration: the real model on the real corpus
# ===========================================================================
def model_or_skip_reason() -> tuple[EmbeddingModel | None, str]:
    """Load the real model once, or explain why the integration tier cannot run."""
    if not sentence_transformers_available():
        return None, "sentence-transformers is not installed (pip install -r requirements-rag.txt)"
    model = EmbeddingModel(EmbeddingConfig())
    try:
        model.load()
    except ModelUnavailableError as exc:
        return None, f"{DEFAULT_MODEL} could not be loaded: {str(exc)[:110]}"
    return model, ""


def test_integration() -> None:
    print(f"\nIntegration -- {DEFAULT_MODEL} (optional)")

    model, reason = model_or_skip_reason()
    if model is None:
        for label in (
            "the real corpus embeds end to end",
            "every chunk produces exactly one embedding",
            "embedding dimensions are consistent",
            "the dimension is the model's documented width",
            "the configured model name is recorded",
            "normalised vectors have approximately unit L2 norm",
            "provenance survives real embedding",
            "chunk ids stay with the right vectors",
            "different chunks get different vectors",
            "batch size does not change real output",
            "single and batch embedding agree within tolerance",
            "repeated embedding is stable within tolerance",
            "the real model is loaded once and reused",
        ):
            skip(label, reason)
        return

    chunks = chunk_corpus()

    # The corpus is embedded exactly once; every check below reuses this.
    results = model.embed_chunks(chunks)

    check("the real corpus embeds end to end", len(results) > 0)
    check("every chunk produces exactly one embedding", len(results) == len(chunks),
          f"{len(results)} embeddings for {len(chunks)} chunks")
    check("embedding dimensions are consistent",
          len({r.dimension for r in results}) == 1, str({r.dimension for r in results}))
    check("the dimension is the model's documented width",
          results[0].dimension == DEFAULT_DIMENSION, str(results[0].dimension))
    check("the configured model name is recorded",
          all(r.model_name == DEFAULT_MODEL for r in results))

    norms = [l2_norm(r.vector) for r in results]
    check("normalised vectors have approximately unit L2 norm",
          all(abs(n - 1.0) <= 1e-4 for n in norms),
          f"range {min(norms):.6f} .. {max(norms):.6f}")

    by_id = {c.chunk_id: c for c in chunks}
    check("provenance survives real embedding",
          all(r.document_id == by_id[r.chunk_id].document_id
              and r.section == by_id[r.chunk_id].section
              and r.content_sha256 == by_id[r.chunk_id].content_sha256
              for r in results))
    check("chunk ids stay with the right vectors",
          [r.chunk_id for r in results] == [c.chunk_id for c in chunks])
    check("different chunks get different vectors",
          len({r.vector for r in results}) == len(results))

    # Small samples from here on, so the corpus is not re-embedded.
    sample = chunks[:3]
    baseline = [r for r in results if r.chunk_id in {c.chunk_id for c in sample}]

    batched = EmbeddingModel(EmbeddingConfig(batch_size=1))
    batched._encoder = model._encoder  # reuse the loaded model, do not load twice
    one_at_a_time = batched.embed_chunks(sample)
    check("batch size does not change real output",
          [r.chunk_id for r in one_at_a_time] == [c.chunk_id for c in sample])
    check("single and batch embedding agree within tolerance",
          all(close_enough(a.vector, b.vector)
              for a, b in zip(one_at_a_time, baseline)),
          f"tolerance {NUMERIC_TOLERANCE}")

    repeat = model.embed_chunks(sample)
    check("repeated embedding is stable within tolerance",
          all(close_enough(a.vector, b.vector) for a, b in zip(repeat, baseline)),
          f"tolerance {NUMERIC_TOLERANCE}")

    check("the real model is loaded once and reused", model.load_count == 1,
          str(model.load_count))

    stats = embedding_statistics(results)
    print(f"        model={stats['model']} dim={stats['dimension']} "
          f"chunks={stats['embeddings']} norms={stats['min_norm']}..{stats['max_norm']}")


# ===========================================================================
def main() -> int:
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print("RAG step 3 -- embeddings")
    print(f"sentence-transformers installed: {sentence_transformers_available()}")

    test_configuration()
    test_embedding_input()
    test_batching_with_stub()
    test_normalisation_with_stub()
    test_provenance_with_stub()
    test_contract_with_stub()
    test_isolation()
    test_load_errors()
    test_adapter_plumbing()
    test_integration()

    total = _passed + _failed
    suffix = f", {_skipped} skipped" if _skipped else ""
    print(f"\n{_passed}/{total} checks passed{suffix}")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
