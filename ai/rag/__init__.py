"""Retrieval-augmented generation support for the AI analysis layer.

Scope of this package
---------------------
``ai.rag`` turns a small, hand-authored, version-controlled corpus of network
security notes into context that can be handed to the existing LLM layer.  It
is a **consumer** of :mod:`ai.schemas`, exactly as :mod:`ai` is a consumer of
:mod:`dpi`.  The dependency runs one way only::

    dpi/  ──────────►  ai/  ──────────►  ai/rag/

Nothing in :mod:`dpi` or in the existing :mod:`ai` modules imports this
package.  Deleting ``ai/rag/`` and ``knowledge/`` leaves the engine, the AI
layer and both existing test suites working unchanged.

Build status
------------
Only **step 1** of the RAG plan is implemented: the knowledge corpus and its
loader.  Chunking, embeddings, the vector store, retrieval, signal extraction,
prompt integration and evaluation are deliberately absent.

Importing this package has **no side effects** and pulls in nothing beyond the
standard library and :mod:`pydantic`, which the project already requires.  In
particular it does not import — or need — numpy, sentence-transformers, torch,
an embedding model, or an API key.

Re-exports are resolved lazily through :pep:`562`.  ``from ai.rag import
load_corpus`` works exactly as if the names were imported eagerly, but
``import ai.rag`` on its own touches no submodule.  That keeps the package
import cheap once the heavier steps land, and it keeps
``python -m ai.rag.documents`` free of the double-import warning ``runpy``
emits when a package ``__init__`` has already imported the module being run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers and editors only
    from .documents import (
        CATEGORY_DIRECTORIES,
        KNOWN_SIGNALS,
        REQUIRED_SECTIONS,
        Category,
        DuplicateIdError,
        FrontMatterError,
        KnowledgeDocument,
        KnowledgeError,
        KnowledgeMetadata,
        MetadataError,
        SectionError,
        default_knowledge_root,
        discover_documents,
        load_corpus,
        load_document,
        parse_document,
        parse_front_matter,
        split_sections,
    )

__all__ = [
    "CATEGORY_DIRECTORIES",
    "KNOWN_SIGNALS",
    "REQUIRED_SECTIONS",
    "Category",
    "DuplicateIdError",
    "FrontMatterError",
    "KnowledgeDocument",
    "KnowledgeError",
    "KnowledgeMetadata",
    "MetadataError",
    "SectionError",
    "default_knowledge_root",
    "discover_documents",
    "load_corpus",
    "load_document",
    "parse_document",
    "parse_front_matter",
    "split_sections",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Resolve a public name from :mod:`ai.rag.documents` on first use."""
    if name in __all__:
        from . import documents

        return getattr(documents, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, "__version__"])
