"""Capture report in, reference knowledge out -- or a clear reason why not.

This is the coordinating layer for steps 1-6::

    CaptureReport -> signals -> queries -> retrieval -> KnowledgeContext

and it is the only place the RAG stages are wired together.  Everything it
depends on is optional: the corpus can be absent, ``numpy`` can be missing,
``sentence-transformers`` can be uninstalled, the model can be uncached and
unreachable.  None of that is an error condition for this project, because the
DPI analysis has already finished before any of it runs.

Never raises, never pretends
----------------------------
:meth:`KnowledgePipeline.build_context` returns a :class:`RAGOutcome` for every
path, successful or not.  It does not raise, so a caller cannot forget to
handle a failure; and it does not fabricate, so a caller cannot mistake a
failure for a thin result.  ``status`` says which happened and ``detail`` says
why, in a sentence meant for a person.

Why every import is deferred
----------------------------
Nothing from :mod:`ai.rag` is imported at module level.  ``numpy`` is required
by the vector store and ``sentence-transformers`` by the embedder, and both
live in ``requirements-rag.txt`` rather than the base requirements -- so
importing them here would turn an optional dependency into a mandatory one and
break ``import ai.rag.pipeline`` on a machine that only wanted the DPI engine.
The imports happen inside :meth:`prepare`, where an ``ImportError`` is a
status rather than a crash.

State lives on the instance
---------------------------
The corpus, the index and the loaded model are cached on the pipeline object,
not in module globals: building them costs a model load and a few hundred
milliseconds, so they should be reused, but a hidden global would make two
different configurations silently share one index.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from ..schemas import CaptureReport
    from .context import KnowledgeContext, KnowledgeContextConfig
    from .retrieval import RetrievalConfig, RetrievalReport
    from .signals import SignalConfig, SignalReport

__all__ = [
    "KnowledgePipeline",
    "RAGOutcome",
    "RAGStatus",
    "default_pipeline",
]


class RAGStatus(str, Enum):
    """What happened when knowledge was requested.

    Every value except :attr:`USED` means the analysis continues *without*
    reference knowledge -- which is a supported outcome, not a degraded one.
    """

    #: Knowledge was retrieved and will be supplied to the model.
    USED = "used"
    #: The caller did not ask for retrieval.
    DISABLED = "disabled"
    #: numpy or sentence-transformers is not installed.
    DEPENDENCY_MISSING = "dependency_missing"
    #: The embedding model could not be loaded or downloaded.
    MODEL_UNAVAILABLE = "model_unavailable"
    #: The knowledge corpus is missing or failed validation.
    CORPUS_UNAVAILABLE = "corpus_unavailable"
    #: Chunking, embedding or indexing failed.
    INDEX_FAILED = "index_failed"
    #: Signal extraction or the search itself failed.
    RETRIEVAL_FAILED = "retrieval_failed"
    #: Everything worked; nothing matched.  Not a failure.
    NO_KNOWLEDGE = "no_knowledge"


@dataclass(frozen=True, slots=True)
class RAGOutcome:
    """The result of asking for knowledge about one capture."""

    status: RAGStatus
    context: "KnowledgeContext | None" = None
    detail: str = ""
    #: Signal types that fired, for the report.  Empty when signals never ran.
    signal_types: tuple[str, ...] = ()
    #: Chunks retrieved before the prompt-side cap was applied.
    retrieved_count: int = 0
    #: Kept for callers that want to show the queries or the raw ranking.
    signal_report: "SignalReport | None" = None
    retrieval_report: "RetrievalReport | None" = None

    @property
    def used(self) -> bool:
        return self.status is RAGStatus.USED

    @property
    def refs(self) -> tuple[str, ...]:
        """Labels supplied to the model; empty unless knowledge was used."""
        return self.context.refs() if self.context is not None else ()

    def knowledge_text(self) -> str | None:
        """The rendered block, or ``None`` when there is nothing to supply."""
        if self.context is None or not self.context.items:
            return None
        return self.context.text

    def describe(self) -> str:
        """One line explaining the status, for a report or the console."""
        if self.status is RAGStatus.USED:
            count = len(self.context.items) if self.context else 0
            return f"{count} reference excerpt(s) supplied to the model."
        return self.detail or _DEFAULT_DETAIL.get(self.status, "Knowledge was not used.")


_DEFAULT_DETAIL: Final[dict[RAGStatus, str]] = {
    RAGStatus.DISABLED: "Retrieval was disabled for this run.",
    RAGStatus.DEPENDENCY_MISSING: (
        "The optional RAG dependencies are not installed. "
        "Install them with: pip install -r requirements-rag.txt"
    ),
    RAGStatus.MODEL_UNAVAILABLE: (
        "The embedding model could not be loaded. Check the model cache and "
        "network, or set DPI_EMBED_OFFLINE=0 to allow a first download."
    ),
    RAGStatus.CORPUS_UNAVAILABLE: "The knowledge corpus could not be loaded.",
    RAGStatus.INDEX_FAILED: "The knowledge index could not be built.",
    RAGStatus.RETRIEVAL_FAILED: "Retrieval failed for this capture.",
    RAGStatus.NO_KNOWLEDGE: "No reference knowledge matched this capture.",
}


class KnowledgePipeline:
    """Builds a knowledge index once, then answers capture reports with it.

    Construct one, reuse it.  :meth:`prepare` is idempotent and is called
    automatically by :meth:`build_context`, so a caller that just wants
    knowledge never has to think about the index at all.
    """

    __slots__ = ("_embedding_config", "_retrieval_config", "_context_config",
                 "_signal_config", "_knowledge_root", "_store", "_embedder",
                 "_status", "_detail", "_chunk_count")

    def __init__(
        self,
        embedding_config: Any = None,
        retrieval_config: "RetrievalConfig | None" = None,
        context_config: "KnowledgeContextConfig | None" = None,
        signal_config: "SignalConfig | None" = None,
        knowledge_root: str | Path | None = None,
    ) -> None:
        self._embedding_config = embedding_config
        self._retrieval_config = retrieval_config
        self._context_config = context_config
        self._signal_config = signal_config
        self._knowledge_root = Path(knowledge_root) if knowledge_root else None
        self._store: Any = None
        self._embedder: Any = None
        self._status: RAGStatus | None = None
        self._detail: str = ""
        self._chunk_count: int = 0

    @classmethod
    def from_index(
        cls,
        store: Any,
        embedder: Any,
        retrieval_config: "RetrievalConfig | None" = None,
        context_config: "KnowledgeContextConfig | None" = None,
        signal_config: "SignalConfig | None" = None,
    ) -> KnowledgePipeline:
        """Wrap an index that has already been built.

        For a caller that wants to build the corpus index once and share it --
        and for tests, which use it to exercise the whole path with a stub
        encoder and no model download.  :meth:`prepare` then has nothing to do.
        """
        pipeline = cls(
            retrieval_config=retrieval_config,
            context_config=context_config,
            signal_config=signal_config,
        )
        pipeline._store = store
        pipeline._embedder = embedder
        pipeline._chunk_count = store.count()
        return pipeline

    # -- state --------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._store is not None and self._embedder is not None

    @property
    def chunk_count(self) -> int:
        """Chunks in the index, or 0 before it is built."""
        return self._chunk_count

    # -- preparation --------------------------------------------------------
    def prepare(self) -> RAGStatus:
        """Load the corpus, embed it and build the index.

        Idempotent, and safe to call on a machine with none of the optional
        dependencies: every failure becomes a :class:`RAGStatus` rather than an
        exception.  A previous failure is remembered, so a missing model is not
        retried once per capture.
        """
        if self.ready:
            return RAGStatus.USED
        if self._status is not None:
            return self._status

        # -- optional dependencies -----------------------------------------
        try:
            from .chunking import chunk_corpus
            from .documents import KnowledgeError, load_corpus
            from .embeddings import (
                EmbeddingConfig,
                EmbeddingModel,
                ModelUnavailableError,
            )
            from .vector_store import VectorRecord, VectorStore
        except ImportError as exc:
            return self._fail(RAGStatus.DEPENDENCY_MISSING,
                              f"{_DEFAULT_DETAIL[RAGStatus.DEPENDENCY_MISSING]} ({exc})")

        # -- corpus ---------------------------------------------------------
        try:
            corpus = load_corpus(self._knowledge_root)
            chunks = chunk_corpus(corpus)
        except KnowledgeError as exc:
            return self._fail(RAGStatus.CORPUS_UNAVAILABLE,
                              f"The knowledge corpus could not be loaded: {exc}")
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            return self._fail(RAGStatus.CORPUS_UNAVAILABLE,
                              f"The knowledge corpus could not be prepared: "
                              f"{type(exc).__name__}: {exc}")

        if not chunks:
            return self._fail(RAGStatus.CORPUS_UNAVAILABLE,
                              "The knowledge corpus is empty.")

        # -- model ----------------------------------------------------------
        embedder = self._embedder
        if embedder is None:
            config = self._embedding_config or EmbeddingConfig()
            embedder = EmbeddingModel(config)
        try:
            embedder.load()
        except ModelUnavailableError as exc:
            return self._fail(RAGStatus.MODEL_UNAVAILABLE, str(exc))

        # -- index ----------------------------------------------------------
        try:
            embeddings = embedder.embed_chunks(list(chunks))
            store = VectorStore("knowledge")
            store.add_many([VectorRecord(chunk=chunk, embedding=embedding)
                            for chunk, embedding in zip(chunks, embeddings)])
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            return self._fail(RAGStatus.INDEX_FAILED,
                              f"The knowledge index could not be built: "
                              f"{type(exc).__name__}: {exc}")

        self._embedder = embedder
        self._store = store
        self._chunk_count = store.count()
        self._status = None
        self._detail = ""
        return RAGStatus.USED

    def _fail(self, status: RAGStatus, detail: str) -> RAGStatus:
        self._status = status
        self._detail = detail
        return status

    # -- use ----------------------------------------------------------------
    def build_context(self, report: "CaptureReport") -> RAGOutcome:
        """Extract signals from ``report``, retrieve, and render the block.

        Returns a :class:`RAGOutcome` in every case.  Signal extraction runs
        even when the index is unavailable, so a caller can still report what
        the capture contained -- the signals are deterministic and need no
        model.
        """
        from .signals import extract_signals

        try:
            signal_report = extract_signals(report, self._signal_config)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            return RAGOutcome(
                status=RAGStatus.RETRIEVAL_FAILED,
                detail=f"Signal extraction failed: {type(exc).__name__}: {exc}",
            )

        signal_types = signal_report.types()

        status = self.prepare()
        if status is not RAGStatus.USED:
            return RAGOutcome(status=status, detail=self._detail,
                              signal_types=signal_types,
                              signal_report=signal_report)

        from .context import build_knowledge_context
        from .retrieval import RetrievalError, retrieve_for_signals

        try:
            retrieval = retrieve_for_signals(signal_report, self._store,
                                             self._embedder, self._retrieval_config)
        except RetrievalError as exc:
            return RAGOutcome(status=RAGStatus.RETRIEVAL_FAILED, detail=str(exc),
                              signal_types=signal_types, signal_report=signal_report)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            return RAGOutcome(
                status=RAGStatus.RETRIEVAL_FAILED,
                detail=f"Retrieval failed: {type(exc).__name__}: {exc}",
                signal_types=signal_types, signal_report=signal_report)

        context = build_knowledge_context(retrieval, self._context_config)

        if not context.items:
            return RAGOutcome(
                status=RAGStatus.NO_KNOWLEDGE,
                context=context,
                detail=_DEFAULT_DETAIL[RAGStatus.NO_KNOWLEDGE],
                signal_types=signal_types,
                signal_report=signal_report,
                retrieval_report=retrieval,
            )

        return RAGOutcome(
            status=RAGStatus.USED,
            context=context,
            detail="",
            signal_types=signal_types,
            retrieved_count=retrieval.chunk_count,
            signal_report=signal_report,
            retrieval_report=retrieval,
        )


def default_pipeline(**overrides: Any) -> KnowledgePipeline:
    """A pipeline with project defaults.

    A function rather than a module-level instance, so nothing is built at
    import time and two callers cannot accidentally share one index.
    """
    return KnowledgePipeline(**overrides)
