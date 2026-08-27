"""Local, free, deterministic embedding of knowledge chunks.

What this does
--------------
Turns :class:`~ai.rag.chunking.KnowledgeChunk` objects into fixed-length float
vectors, using a small sentence-transformer model that runs **entirely on this
machine**.  No API key, no Groq, no OpenAI, no network once the model has been
downloaded once.  The provider abstraction in :mod:`ai.llm_client` is not
involved at any point: text generation and embedding are separate concerns
with separate models, and this module deliberately shares nothing with it.

What this does *not* do: store vectors, index them, compare them, search them
or retrieve anything.  There is no similarity function here, not even a dot
product.  A vector store is step 4 and retrieval is step 6.

Why the embedding is local
--------------------------
Groq -- the project's LLM provider -- serves no embedding endpoint at all, so
a hosted embedding would mean adding a second vendor and a second bill.  A
384-dimension model that runs on a CPU in a few hundred milliseconds makes
that unnecessary, and it gives the project a property worth stating plainly:
**retrieval never leaves the machine.**  Only the final assembled prompt is
sent to a provider.

Model asymmetry (relevant later, not now)
-----------------------------------------
``bge`` models are trained asymmetrically: passages are embedded as-is, while
*queries* are expected to carry the instruction prefix ``"Represent this
sentence for searching relevant passages: "``.  Step 3 embeds passages only,
so no prefix is applied and none is needed.  The query side belongs to
retrieval (step 6) and is deliberately absent here -- but it is worth knowing
that omitting the prefix at that point silently costs recall, which is the
kind of bug that never raises an exception.

Determinism
-----------
Preprocessing is fully deterministic: the same chunk always produces the same
input string, and therefore the same ``input_sha256``.  The *numeric* output
is deterministic in practice for a fixed model, version and device, but
floating-point arithmetic is not contractually reproducible across BLAS
builds, thread counts or hardware, so this module documents a tolerance
(:data:`NUMERIC_TOLERANCE`) rather than promising bit-exact equality.

No fallback, ever
-----------------
If the model cannot be loaded, this module raises
:class:`ModelUnavailableError` and stops.  It does not substitute a different
model, and it does not invent vectors.  Silent degradation in an embedding
layer is invisible: hashed or random vectors would produce a perfectly
functional index that retrieves nonsense.

The ``encoder`` seam on :class:`EmbeddingModel` exists so tests can exercise
batching, ordering and normalisation logic without downloading a model.  It is
never selected automatically -- nothing constructs one on your behalf, and
leaving it unset means loading the real model or failing loudly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from typing import Final, Iterator, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .chunking import KnowledgeChunk
from .documents import Category

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DIMENSION",
    "DEFAULT_MODEL",
    "ENV_BATCH_SIZE",
    "ENV_CACHE_FOLDER",
    "ENV_DEVICE",
    "ENV_MODEL",
    "ENV_OFFLINE",
    "NUMERIC_TOLERANCE_NOTE",
    "NUMERIC_TOLERANCE",
    "EMBEDDING_INPUT_TEMPLATE",
    "EmbeddingConfig",
    "EmbeddingError",
    "EmbeddingInputError",
    "EmbeddingModel",
    "EmbeddingResult",
    "Encoder",
    "ModelUnavailableError",
    "build_embedding_input",
    "embedding_statistics",
    "l2_norm",
    "serialize_embedding_metadata",
    "sentence_transformers_available",
]


# ===========================================================================
# Defaults and environment
# ===========================================================================
#: The default embedding model.
#:
#: ``BAAI/bge-small-en-v1.5`` is Apache-licensed, about 130 MB, runs
#: comfortably on a CPU, and produces 384-dimension vectors with a 512-token
#: window.  That window is the operative reason it was chosen over
#: ``all-MiniLM-L6-v2``, which is smaller and faster but truncates at **256
#: tokens** -- silently, with no error -- and would therefore lose the tail of
#: the larger chunks this corpus produces.
DEFAULT_MODEL: Final[str] = "BAAI/bge-small-en-v1.5"

#: Vector width the default model produces.  Documentation and a test
#: expectation only: the code always reads the real dimension from the loaded
#: model and never assumes this value.
DEFAULT_DIMENSION: Final[int] = 384

#: Chunks per encode() call.  Small because the corpus is small and CPU
#: memory is the constraint that actually bites; raise it for a GPU.
DEFAULT_BATCH_SIZE: Final[int] = 16

#: Tolerance used when comparing two embeddings of the same text.
#:
#: Floating-point reduction order is not guaranteed across BLAS backends,
#: thread counts or devices, so re-embedding identical text can differ in the
#: last few bits.  1e-5 per component is comfortably tighter than any
#: difference that would affect ranking, and comfortably looser than the noise
#: floor of float32 accumulation.
NUMERIC_TOLERANCE: Final[float] = 1e-5

NUMERIC_TOLERANCE_NOTE: Final[str] = (
    "Embeddings are compared component-wise within NUMERIC_TOLERANCE rather "
    "than for exact equality: float32 reductions are not bit-reproducible "
    "across BLAS builds, thread counts or devices."
)

ENV_MODEL: Final[str] = "DPI_EMBED_MODEL"
ENV_BATCH_SIZE: Final[str] = "DPI_EMBED_BATCH_SIZE"
ENV_DEVICE: Final[str] = "DPI_EMBED_DEVICE"
ENV_OFFLINE: Final[str] = "DPI_EMBED_OFFLINE"
ENV_CACHE_FOLDER: Final[str] = "DPI_EMBED_CACHE"

#: A model name is either ``org/name`` from the hub or a local directory path.
_MODEL_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[\w./\\:-]+$")


# ===========================================================================
# Errors
# ===========================================================================
class EmbeddingError(Exception):
    """Base class for embedding problems."""


class ModelUnavailableError(EmbeddingError):
    """The embedding model could not be loaded.

    Raised when the library is missing, the model is not cached and cannot be
    fetched, or the name is wrong.  Never followed by a fallback: an embedding
    layer that quietly substitutes something else produces an index that looks
    healthy and retrieves nonsense.
    """


class EmbeddingInputError(EmbeddingError):
    """The input to be embedded is empty, or is not a valid chunk."""


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Embedding parameters.  Immutable, so a run cannot drift mid-corpus."""

    model_name: str = DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    normalize: bool = True
    device: str | None = None
    #: Load only from the local cache; never contact the model hub.
    local_files_only: bool = False
    #: Override the model cache directory.
    cache_folder: str | None = None

    def __post_init__(self) -> None:
        name = self.model_name.strip() if isinstance(self.model_name, str) else ""
        if not name:
            raise ValueError("model_name must not be empty")
        if name != self.model_name:
            raise ValueError("model_name must not have leading or trailing whitespace")
        if not _MODEL_NAME_PATTERN.match(name):
            raise ValueError(
                f"model_name {self.model_name!r} is not a valid hub id or path; "
                'expected something like "BAAI/bge-small-en-v1.5"'
            )
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise ValueError("batch_size must be an integer")
        if not 1 <= self.batch_size <= 1024:
            raise ValueError(f"batch_size must be between 1 and 1024 (got {self.batch_size})")
        if self.device is not None and not str(self.device).strip():
            raise ValueError("device must be None or a non-empty string")

    @classmethod
    def from_env(cls, **overrides: object) -> EmbeddingConfig:
        """Build a config from the environment, with explicit overrides winning.

        Nothing here is a secret -- these are a model name, a batch size and a
        device -- so unlike :class:`~ai.config.AIConfig` there is no masking to
        do and no key to leak.
        """
        raw_batch = os.environ.get(ENV_BATCH_SIZE)
        try:
            batch_size = int(raw_batch) if raw_batch else DEFAULT_BATCH_SIZE
        except ValueError as exc:
            raise ValueError(
                f"{ENV_BATCH_SIZE}={raw_batch!r} is not an integer"
            ) from exc

        values: dict[str, object] = {
            "model_name": os.environ.get(ENV_MODEL, DEFAULT_MODEL),
            "batch_size": batch_size,
            "device": os.environ.get(ENV_DEVICE) or None,
            "local_files_only": bool(os.environ.get(ENV_OFFLINE)),
            "cache_folder": os.environ.get(ENV_CACHE_FOLDER) or None,
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]


# ===========================================================================
# Embedding input
# ===========================================================================
#: Template for the text actually sent to the model.
#:
#: Three provenance lines then the chunk body.  Embedding the bare body loses
#: information the retriever needs: a section headed "Indicators" is nearly
#: meaningless without knowing indicators *of what*, and a bulleted list of
#: benign explanations reads almost identically across documents.  Prefixing
#: the title, category and section makes each chunk self-describing in vector
#: space at a cost of roughly twenty tokens.
#:
#: What is deliberately **excluded**, and why:
#:
#: * ``keywords`` -- lexical handles meant for the hybrid search planned for
#:   week 4; embedding them would double-count the same words.
#: * ``mitre``, ``licence``, ``sources``, ``version`` -- identifiers and
#:   boilerplate that carry no topical meaning and would pull unrelated
#:   documents together by their shared licence string.
#: * anything runtime -- no timestamps, no addresses, no packet data, no keys,
#:   no environment values.  This module embeds the corpus and nothing else,
#:   and the corpus is static, reviewed text.
EMBEDDING_INPUT_TEMPLATE: Final[str] = "Title: {title}\nCategory: {category}\nSection: {section}\n\n{body}"


def build_embedding_input(chunk: KnowledgeChunk) -> str:
    """Return the exact string that will be embedded for ``chunk``.

    Pure and deterministic: same chunk, same string, always.  Separated from
    the model so it can be tested -- and inspected -- with nothing installed.

    The chunk's full ``text`` is used, including any overlap it carries.  The
    overlap exists precisely so a split chunk keeps the context that gives it
    its subject; dropping it before embedding would discard the thing it was
    added for.
    """
    if not isinstance(chunk, KnowledgeChunk):
        raise EmbeddingInputError(
            f"expected a KnowledgeChunk, got {type(chunk).__name__}; "
            "chunks must come from ai.rag.chunking, already validated"
        )

    category = chunk.category.value if isinstance(chunk.category, Category) else str(chunk.category)
    text = EMBEDDING_INPUT_TEMPLATE.format(
        title=chunk.title,
        category=category,
        section=chunk.section,
        body=chunk.text,
    )
    if not text.strip():  # pragma: no cover - the chunk model forbids empty text
        raise EmbeddingInputError(f"chunk {chunk.chunk_id} produced an empty input")
    return text


def _input_hash(text: str) -> str:
    """SHA-256 of an embedding input, for cache keys and change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ===========================================================================
# Result
# ===========================================================================
def l2_norm(vector: Sequence[float]) -> float:
    """Euclidean length of a vector.

    Written out rather than imported so this module needs no numeric library
    of its own; ``fsum`` keeps the accumulation stable for 384 terms.
    """
    return sqrt(fsum(component * component for component in vector))


class EmbeddingResult(BaseModel):
    """One chunk's vector, with the provenance needed to cite it later.

    Provenance travels *with* the vector rather than beside it, so a later
    index can be rebuilt, inspected or debugged without re-reading the corpus,
    and a retrieval hit can name its document and section immediately.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # -- provenance ---------------------------------------------------------
    chunk_id: str = Field(min_length=3)
    document_id: str = Field(min_length=3)
    category: Category
    section: str = Field(min_length=1)
    heading_path: str = Field(min_length=1)
    content_sha256: str = Field(min_length=64, max_length=64)
    input_sha256: str = Field(
        min_length=64,
        max_length=64,
        description="Hash of the exact string embedded, not of the chunk text.",
    )

    # -- embedding ----------------------------------------------------------
    model_name: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    normalized: bool = Field(
        description="Whether the vector was L2-normalised. Recorded, never assumed."
    )
    vector: tuple[float, ...] = Field(min_length=1)

    @field_validator("vector")
    @classmethod
    def _finite(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if not all(isfinite(component) for component in v):
            raise ValueError("vector contains NaN or infinity")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> EmbeddingResult:
        if len(self.vector) != self.dimension:
            raise ValueError(
                f"dimension is {self.dimension} but the vector has {len(self.vector)} components"
            )
        if self.normalized and abs(l2_norm(self.vector) - 1.0) > 1e-3:
            raise ValueError(
                f"vector is marked normalized but its L2 norm is {l2_norm(self.vector):.6f}"
            )
        return self

    def metadata(self) -> dict[str, object]:
        """Everything except the vector -- for logs, manifests and diffs."""
        return self.model_dump(mode="json", exclude={"vector"})


# ===========================================================================
# Encoder seam
# ===========================================================================
@runtime_checkable
class Encoder(Protocol):
    """The minimal thing :class:`EmbeddingModel` needs from a model.

    Deliberately narrower than ``SentenceTransformer``: a name, a width, and
    one method that turns strings into vectors.  Keeping the surface this
    small is what lets the batching, ordering and normalisation logic be
    tested without a 2 GB dependency, and what would let a different backend
    (ONNX Runtime, say) drop in later without touching anything else.
    """

    @property
    def name(self) -> str: ...

    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str], normalize: bool) -> list[list[float]]: ...


class _SentenceTransformerEncoder:
    """Adapter over ``sentence_transformers.SentenceTransformer``."""

    __slots__ = ("_model", "_name")

    def __init__(self, model: object, name: str) -> None:
        self._model = model
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def dimension(self) -> int:
        # sentence-transformers renamed this in v6; support both spellings so
        # the adapter does not emit a deprecation warning on new releases or
        # break on older ones.
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        if getter is None:  # pragma: no cover - not a sentence-transformer
            raise ModelUnavailableError(
                f"model {self._name!r} does not expose an embedding dimension"
            )
        size = getter()
        if not size:  # pragma: no cover - a model without a reported width
            raise ModelUnavailableError(
                f"model {self._name!r} did not report an embedding dimension"
            )
        return int(size)

    def encode(self, texts: Sequence[str], normalize: bool) -> list[list[float]]:
        vectors = self._model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=len(texts),
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(component) for component in row] for row in vectors]


def sentence_transformers_available() -> bool:
    """Whether the library is importable.  Says nothing about the model itself."""
    try:
        import sentence_transformers  # noqa: F401
    except Exception:  # noqa: BLE001 - a broken install is also unavailable
        return False
    return True


# ===========================================================================
# The model
# ===========================================================================
class EmbeddingModel:
    """Loads one embedding model once and embeds chunks with it.

    The model is loaded lazily on first use and then reused for the life of
    the instance -- :attr:`load_count` makes that testable.  Loading a
    sentence-transformer costs on the order of a second and hundreds of
    megabytes of RAM, so reloading per chunk would dominate the runtime of
    every later step.
    """

    __slots__ = ("_config", "_encoder", "_dimension", "_load_count")

    def __init__(self, config: EmbeddingConfig | None = None, encoder: Encoder | None = None):
        self._config = config or EmbeddingConfig()
        self._encoder = encoder
        self._dimension: int | None = None
        self._load_count = 0
        if encoder is not None:
            self._load_count = 1

    # -- state --------------------------------------------------------------
    @property
    def config(self) -> EmbeddingConfig:
        return self._config

    @property
    def loaded(self) -> bool:
        return self._encoder is not None

    @property
    def load_count(self) -> int:
        """How many times a model has actually been constructed.  Should be 1."""
        return self._load_count

    @property
    def model_name(self) -> str:
        return self._config.model_name

    # -- loading ------------------------------------------------------------
    def load(self) -> None:
        """Load the model if it is not loaded already.

        Raises :class:`ModelUnavailableError` with an actionable message.  No
        alternative model is ever substituted.
        """
        if self._encoder is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ModelUnavailableError(
                "sentence-transformers is not installed. "
                "Install the optional embedding dependency with: "
                "pip install -r requirements-rag.txt"
            ) from exc

        kwargs: dict[str, object] = {}
        if self._config.device:
            kwargs["device"] = self._config.device
        if self._config.cache_folder:
            kwargs["cache_folder"] = self._config.cache_folder
        if self._config.local_files_only:
            kwargs["local_files_only"] = True

        try:
            model = SentenceTransformer(self._config.model_name, **kwargs)
        except TypeError:
            # Older releases have no local_files_only parameter; the hub reads
            # the same intent from the environment.
            kwargs.pop("local_files_only", None)
            previous = os.environ.get("HF_HUB_OFFLINE")
            if self._config.local_files_only:
                os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                model = SentenceTransformer(self._config.model_name, **kwargs)
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                raise self._load_failure(exc) from exc
            finally:
                if self._config.local_files_only:
                    if previous is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = previous
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            raise self._load_failure(exc) from exc

        self._encoder = _SentenceTransformerEncoder(model, self._config.model_name)
        self._dimension = None
        self._load_count += 1

    def _load_failure(self, exc: Exception) -> ModelUnavailableError:
        hint = (
            "the model is not in the local cache and offline loading was requested"
            if self._config.local_files_only
            else "check the model name, the network, and the model cache"
        )
        return ModelUnavailableError(
            f"could not load embedding model {self._config.model_name!r}: "
            f"{type(exc).__name__}: {exc}. Hint: {hint}. "
            "No substitute model is used."
        )

    def dimension(self) -> int:
        """Vector width of the loaded model, read from the model itself."""
        self.load()
        if self._dimension is None:
            assert self._encoder is not None
            self._dimension = self._encoder.dimension()
        return self._dimension

    # -- embedding ----------------------------------------------------------
    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Embed raw strings, in order, in batches.

        The Nth output corresponds to the Nth input.  Batching is explicit
        here rather than delegated, so the boundary between batches is visible
        and testable, and so a future progress callback has somewhere to live.
        """
        items = list(texts)
        if not items:
            raise EmbeddingInputError("nothing to embed: the input sequence is empty")
        for index, text in enumerate(items):
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingInputError(f"input {index} is empty or not a string")

        self.load()
        assert self._encoder is not None
        expected = self.dimension()

        vectors: list[tuple[float, ...]] = []
        for batch in _batches(items, self._config.batch_size):
            encoded = self._encoder.encode(batch, self._config.normalize)
            if len(encoded) != len(batch):
                raise EmbeddingError(
                    f"encoder returned {len(encoded)} vectors for {len(batch)} inputs"
                )
            for row in encoded:
                if len(row) != expected:
                    raise EmbeddingError(
                        f"encoder returned a {len(row)}-dimension vector, expected {expected}"
                    )
                vectors.append(tuple(float(component) for component in row))

        return tuple(vectors)

    def embed_chunks(self, chunks: Sequence[KnowledgeChunk]) -> tuple[EmbeddingResult, ...]:
        """Embed chunks, preserving order and provenance.

        Output position N corresponds to input position N, and each result
        carries its own ``chunk_id`` -- so the association survives even if a
        caller later sorts or filters the results.
        """
        items = list(chunks)
        if not items:
            raise EmbeddingInputError("nothing to embed: no chunks were supplied")

        inputs = [build_embedding_input(chunk) for chunk in items]
        vectors = self.embed_texts(inputs)

        results: list[EmbeddingResult] = []
        for chunk, text, vector in zip(items, inputs, vectors):
            results.append(
                EmbeddingResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    category=chunk.category,
                    section=chunk.section,
                    heading_path=chunk.heading_path,
                    content_sha256=chunk.content_sha256,
                    input_sha256=_input_hash(text),
                    model_name=self.model_name,
                    dimension=len(vector),
                    normalized=self._config.normalize,
                    vector=vector,
                )
            )
        return tuple(results)


def _batches(items: Sequence[str], size: int) -> Iterator[list[str]]:
    """Split ``items`` into consecutive batches of at most ``size``.

    Consecutive and in order: batching is a memory strategy, never a
    reordering.
    """
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


# ===========================================================================
# Reporting
# ===========================================================================
def embedding_statistics(results: Sequence[EmbeddingResult]) -> dict[str, object]:
    """Summarise an embedding run.  Never includes vector components."""
    if not results:
        return {"embeddings": 0, "documents": 0, "model": "", "dimension": 0,
                "normalized": False, "min_norm": 0.0, "max_norm": 0.0}

    norms = [l2_norm(r.vector) for r in results]
    return {
        "embeddings": len(results),
        "documents": len({r.document_id for r in results}),
        "model": results[0].model_name,
        "dimension": results[0].dimension,
        "normalized": results[0].normalized,
        "min_norm": round(min(norms), 6),
        "max_norm": round(max(norms), 6),
    }


def serialize_embedding_metadata(results: Sequence[EmbeddingResult]) -> str:
    """Stable JSON of the metadata only -- provenance, model, dimension.

    Vectors are excluded on purpose: this is the human-readable half, meant
    for diffing an index build against the previous one.  Persisting vectors
    is the vector store's job, which is step 4.
    """
    return json.dumps(
        [result.metadata() for result in results],
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )


# ===========================================================================
# Manual check:  python -m ai.rag.embeddings
# ===========================================================================
if __name__ == "__main__":  # pragma: no cover - manual check
    import sys
    import time

    from .chunking import chunk_corpus

    settings = EmbeddingConfig.from_env()
    corpus_chunks = chunk_corpus()

    print(f"model:      {settings.model_name}")
    print(f"chunks:     {len(corpus_chunks)}")
    print(f"batch size: {settings.batch_size}")
    print(f"normalized: {str(settings.normalize).lower()}")
    print(f"offline:    {str(settings.local_files_only).lower()}")

    if not sentence_transformers_available():
        print("\nsentence-transformers is not installed; nothing was embedded.")
        print("Install it with: pip install -r requirements-rag.txt")
        sys.exit(1)

    started = time.monotonic()
    try:
        embedder = EmbeddingModel(settings)
        embedded = embedder.embed_chunks(corpus_chunks)
    except EmbeddingError as error:
        print(f"\nembedding failed\n  {error}")
        sys.exit(1)

    stats = embedding_statistics(embedded)
    print(f"dimension:  {stats['dimension']}")
    print(f"embedded:   {stats['embeddings']} chunks from {stats['documents']} documents "
          f"in {time.monotonic() - started:.1f}s")
    print(f"L2 norms:   {stats['min_norm']} .. {stats['max_norm']}")
    print(f"loads:      {embedder.load_count}")
