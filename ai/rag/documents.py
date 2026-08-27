"""The knowledge corpus: metadata model, front-matter parser and loader.

A knowledge document is a Markdown file under ``knowledge/<category>/`` with a
YAML-style front-matter block and six fixed sections.  Both halves are
validated here, and a document that fails validation stops the load with a
message naming the file and the problem.  Nothing partially-valid is ever
returned: a corpus either loads completely or not at all.

Why validation is this strict
-----------------------------
Everything downstream of this module trusts the corpus.  Chunking (step 2)
splits on the section headings, retrieval (step 6) filters on ``applies_to``,
and the prompt (step 8) cites documents by ``id``.  A missing heading or a
typo'd signal name would surface much later as *silently degraded retrieval* —
the worst kind of RAG bug, because it produces plausible output.  Catching it
at load time turns a quality problem into a build error.

The corpus is also the trust boundary.  Retrieved text is fed to an LLM, so a
document is untrusted input in the same sense a TLS SNI string is.  The
defence is that documents are hand-authored, reviewed in version control, and
schema-checked here.  Content screening for injection patterns belongs to the
ingest step (step 4 of the plan) and is deliberately not implemented yet.

Why a hand-written front-matter parser
--------------------------------------
The project depends on ``pydantic`` and ``openai`` and nothing else.  Adding
PyYAML for a dozen scalars and lists would be a poor trade, and full YAML
brings behaviour this corpus does not want — most notably that ``version: 1.0``
parses as the *float* ``1.0``, so ``1.10`` would silently equal ``1.1``.

:func:`parse_front_matter` therefore accepts one small, explicit subset:

* ``key: value`` — the value is always kept as a **string**
* ``key:`` followed by ``- item`` lines — a list of strings
* ``key: []`` — an empty list
* ``#`` comment lines and blank lines are ignored
* single or double quotes around a scalar are stripped

Anything else — nested mappings, tabs, multi-line scalars, duplicate keys —
raises :class:`FrontMatterError` rather than being guessed at.  Deterministic
and dependency-free beats permissive.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..schemas import Severity

__all__ = [
    "CATEGORY_DIRECTORIES",
    "FRONT_MATTER_DELIMITER",
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


# ===========================================================================
# Vocabulary
# ===========================================================================
class Category(str, Enum):
    """The six knowledge categories.

    Each value is also the directory name under ``knowledge/``, and the
    declaration order is the corpus sort order: reference material first,
    adversary behaviour last.  That ordering is reused when knowledge is
    assembled into a prompt, so ground truth and normal-behaviour context are
    read before attack patterns.
    """

    GLOSSARY = "glossary"
    PROTOCOLS = "protocols"
    BASELINES = "baselines"
    DETECTION_HEURISTICS = "detection-heuristics"
    TRIAGE_PLAYBOOKS = "triage-playbooks"
    ATTACK_PATTERNS = "attack-patterns"


#: Directory name -> category, for checking a document is filed where it says.
CATEGORY_DIRECTORIES: Final[dict[str, Category]] = {c.value: c for c in Category}

#: The six sections every document must contain, in this order.
#:
#: "What the DPI engine can observe" is the section that makes this corpus
#: project-specific rather than generic security writing: it names the actual
#: :class:`~ai.schemas.FlowRecord` and :class:`~ai.schemas.CaptureReport`
#: fields that bear on the topic, so retrieved text connects to real data.
REQUIRED_SECTIONS: Final[tuple[str, ...]] = (
    "Summary",
    "What the DPI engine can observe",
    "Indicators",
    "Benign explanations",
    "Recommended checks",
    "References",
)

#: The reserved signal vocabulary that ``applies_to`` is checked against.
#:
#: Signal extraction itself is step 5 and does not exist yet.  Fixing the names
#: now means the corpus and the future extractor cannot drift apart silently: a
#: document claiming to answer ``dns_beaconing`` — a signal that can never be
#: computed, because :class:`~ai.schemas.FlowRecord` carries no timestamps —
#: is rejected here instead of sitting in the index answering nothing.
#:
#: Adding a signal means adding it to this set first, in the same commit as the
#: document that uses it.
KNOWN_SIGNALS: Final[frozenset[str]] = frozenset(
    {
        # DNS
        "dns_high_volume",
        "dns_high_cardinality",
        "dns_anomalous_label",
        # Scanning
        "scan_port_fanout",
        "scan_half_open",
        # Classification
        "unknown_app_share",
        "tls_without_sni",
        "plaintext_http",
        "quic_present",
        # Volume and destination shape
        "upload_asymmetry",
        "nonstandard_port_egress",
        # Engine decisions
        "blocked_traffic_present",
        # Always emitted, so retrieval is never uniformly alarm-shaped
        "baseline_web_browsing",
    }
)

#: Front-matter fence.
FRONT_MATTER_DELIMITER: Final[str] = "---"

_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d+\.\d+$")
_MITRE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^T\d{4}(?:\.\d{3})?$")
_KEYWORD_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9 ._/-]*$")
_H2_PATTERN: Final[re.Pattern[str]] = re.compile(r"^##\s+(.+?)\s*$")
_SCALAR_PATTERN: Final[re.Pattern[str]] = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(.*)$")
_LIST_ITEM_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\s*-\s+(.*)$")


# ===========================================================================
# Errors
# ===========================================================================
class KnowledgeError(Exception):
    """Base class for every corpus problem.

    Carries the offending path so a caller can report *which* document failed
    without re-deriving it.
    """

    def __init__(self, message: str, source: str | Path | None = None) -> None:
        self.source = str(source) if source is not None else ""
        super().__init__(f"{self.source}: {message}" if self.source else message)


class FrontMatterError(KnowledgeError):
    """The front-matter block is missing, unterminated or not in the subset."""


class MetadataError(KnowledgeError):
    """Front matter parsed, but failed schema validation."""


class SectionError(KnowledgeError):
    """The Markdown body does not match the required section template."""


class DuplicateIdError(KnowledgeError):
    """Two documents declare the same ``id``."""


# ===========================================================================
# Metadata model
# ===========================================================================
class KnowledgeMetadata(BaseModel):
    """Validated front matter for one knowledge document.

    ``extra="forbid"`` is the point of this model as much as the field types
    are: a misspelled key (``keyword:`` for ``keywords:``) is a silent
    retrieval-quality bug, so it is rejected rather than ignored.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        min_length=3,
        max_length=64,
        description="Stable slug. Cited by the model, so it must never churn.",
    )
    title: str = Field(min_length=3, max_length=120)
    category: Category
    version: str = Field(description='Document version as "MAJOR.MINOR", e.g. "1.0".')
    updated: date = Field(description="ISO date of the last substantive edit.")

    applies_to: list[str] = Field(
        min_length=1,
        description="Signal ids this document answers. Checked against KNOWN_SIGNALS.",
    )
    keywords: list[str] = Field(
        min_length=1,
        max_length=20,
        description="Lexical handles for the hybrid retrieval planned in week 4.",
    )

    mitre: list[str] = Field(
        default_factory=list,
        description='MITRE ATT&CK technique ids, e.g. "T1071.004". May be empty.',
    )
    severity_hint: Severity = Field(
        description="Advisory only. The model is told this never binds its own assessment."
    )

    sources: list[str] = Field(
        min_length=1,
        description="Where this content came from. Mandatory: provenance is not optional.",
    )
    licence: str = Field(
        min_length=2,
        max_length=64,
        description='Licence of the content, e.g. "CC-BY-4.0" or "project-authored".',
    )

    # -- validators ---------------------------------------------------------
    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _ID_PATTERN.match(v):
            raise ValueError(
                f"id must be a lowercase hyphenated slug (got {v!r}); "
                "it appears in citations, so it must be URL- and prompt-safe"
            )
        return v

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if not _VERSION_PATTERN.match(v):
            raise ValueError(f'version must look like "1.0" (got {v!r})')
        return v

    @field_validator("applies_to")
    @classmethod
    def _check_signals(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - KNOWN_SIGNALS)
        if unknown:
            raise ValueError(
                f"applies_to names signals that do not exist: {unknown}. "
                f"Known signals: {sorted(KNOWN_SIGNALS)}"
            )
        if len(set(v)) != len(v):
            raise ValueError("applies_to contains duplicates")
        return v

    @field_validator("keywords")
    @classmethod
    def _check_keywords(cls, v: list[str]) -> list[str]:
        for word in v:
            if not _KEYWORD_PATTERN.match(word):
                raise ValueError(
                    f"keyword {word!r} must be lowercase alphanumeric "
                    "(spaces, dots, underscores, slashes and hyphens allowed)"
                )
        if len(set(v)) != len(v):
            raise ValueError("keywords contains duplicates")
        return v

    @field_validator("mitre")
    @classmethod
    def _check_mitre(cls, v: list[str]) -> list[str]:
        for technique in v:
            if not _MITRE_PATTERN.match(technique):
                raise ValueError(
                    f'mitre id {technique!r} must look like "T1071" or "T1071.004"'
                )
        return v

    @field_validator("sources")
    @classmethod
    def _check_sources(cls, v: list[str]) -> list[str]:
        for source in v:
            if not source.strip():
                raise ValueError("sources contains an empty entry")
        return v


# ===========================================================================
# Document
# ===========================================================================
@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    """One loaded, fully validated knowledge document.

    ``sections`` preserves :data:`REQUIRED_SECTIONS` order, which is what the
    chunker in step 2 will iterate; ``sha256`` is the content hash the index
    manifest in step 4 will record so corpus drift is detectable.
    """

    metadata: KnowledgeMetadata
    sections: dict[str, str]
    #: Path relative to the corpus root, POSIX-style, so it is stable across OSes.
    relative_path: str
    #: SHA-256 of the file's raw bytes.
    sha256: str

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def category(self) -> Category:
        return self.metadata.category

    @property
    def title(self) -> str:
        return self.metadata.title

    def section(self, name: str) -> str:
        """Return one section's body text."""
        return self.sections[name]

    def word_count(self) -> int:
        return sum(len(body.split()) for body in self.sections.values())


# ===========================================================================
# Front-matter parsing
# ===========================================================================
def parse_front_matter(
    text: str, source: str | Path | None = None
) -> tuple[dict[str, Any], str]:
    """Split ``text`` into a front-matter mapping and the Markdown body.

    Returns ``(mapping, body)``.  Every scalar is returned as a ``str`` and
    every list as a ``list[str]``; no type coercion happens here, so
    :class:`KnowledgeMetadata` sees exactly what the file says.

    Raises :class:`FrontMatterError` for anything outside the documented
    subset.
    """
    if not text.startswith(FRONT_MATTER_DELIMITER):
        raise FrontMatterError(
            f"document must start with a {FRONT_MATTER_DELIMITER!r} front-matter fence",
            source,
        )

    lines = text.splitlines()
    if lines[0].strip() != FRONT_MATTER_DELIMITER:
        raise FrontMatterError("malformed opening front-matter fence", source)

    closing = next(
        (i for i, line in enumerate(lines[1:], start=1)
         if line.strip() == FRONT_MATTER_DELIMITER),
        None,
    )
    if closing is None:
        raise FrontMatterError(
            f"front matter is never closed by a {FRONT_MATTER_DELIMITER!r} line", source
        )

    mapping: dict[str, Any] = {}
    current_list_key: str | None = None

    for number, raw in enumerate(lines[1:closing], start=2):
        if "\t" in raw:
            raise FrontMatterError(
                f"line {number}: tabs are not allowed in front matter", source
            )

        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # A list item continues the key that opened the list.
        item = _LIST_ITEM_PATTERN.match(raw)
        if item is not None:
            if current_list_key is None:
                raise FrontMatterError(
                    f"line {number}: list item with no key above it", source
                )
            mapping[current_list_key].append(_unquote(item.group(1).strip()))
            continue

        scalar = _SCALAR_PATTERN.match(raw)
        if scalar is None:
            raise FrontMatterError(
                f"line {number}: expected 'key: value' or '- item', got {stripped!r}",
                source,
            )

        key, rest = scalar.group(1), scalar.group(2).strip()
        if key in mapping:
            raise FrontMatterError(f"line {number}: duplicate key {key!r}", source)

        if rest == "":
            # Opens a block list; items follow on subsequent lines.
            mapping[key] = []
            current_list_key = key
        elif rest == "[]":
            mapping[key] = []
            current_list_key = None
        elif rest.startswith("["):
            raise FrontMatterError(
                f"line {number}: inline lists are not supported; use '- item' lines "
                "(only '[]' is accepted, for an empty list)",
                source,
            )
        else:
            mapping[key] = _unquote(rest)
            current_list_key = None

    if not mapping:
        raise FrontMatterError("front matter is empty", source)

    body = "\n".join(lines[closing + 1:])
    return mapping, body


def _unquote(value: str) -> str:
    """Strip one layer of matching quotes, if present."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


# ===========================================================================
# Section parsing
# ===========================================================================
def split_sections(body: str, source: str | Path | None = None) -> dict[str, str]:
    """Split a document body into its ``## `` sections and validate the template.

    Every document must contain exactly :data:`REQUIRED_SECTIONS`, in that
    order, each with non-empty content.  Extra or reordered headings are
    rejected: a uniform template is what lets the chunker treat one section as
    one self-contained, citable unit without any per-document special casing.
    """
    found: list[str] = []
    bodies: dict[str, list[str]] = {}
    current: str | None = None

    for line in body.splitlines():
        heading = _H2_PATTERN.match(line)
        if heading is not None:
            current = heading.group(1).strip()
            if current in bodies:
                raise SectionError(f"duplicate section heading {current!r}", source)
            found.append(current)
            bodies[current] = []
            continue
        if current is not None:
            bodies[current].append(line)

    if found != list(REQUIRED_SECTIONS):
        missing = [s for s in REQUIRED_SECTIONS if s not in found]
        unexpected = [s for s in found if s not in REQUIRED_SECTIONS]
        problems = []
        if missing:
            problems.append(f"missing sections {missing}")
        if unexpected:
            problems.append(f"unexpected sections {unexpected}")
        if not problems:
            problems.append(f"sections are out of order: {found}")
        raise SectionError(
            "; ".join(problems) + f"; required order is {list(REQUIRED_SECTIONS)}",
            source,
        )

    sections: dict[str, str] = {}
    for name in REQUIRED_SECTIONS:
        content = "\n".join(bodies[name]).strip()
        if not content:
            raise SectionError(f"section {name!r} is empty", source)
        sections[name] = content
    return sections


# ===========================================================================
# Loading
# ===========================================================================
def default_knowledge_root() -> Path:
    """Return ``<project root>/knowledge``.

    Derived from this file's location (``ai/rag/documents.py``) rather than
    from the working directory, so a loader call works the same whether the
    project is run from its own directory, from a test runner, or with ``-m``.
    """
    return Path(__file__).resolve().parents[2] / "knowledge"


def parse_document(
    text: str, relative_path: str, source: str | Path | None = None
) -> KnowledgeDocument:
    """Parse and validate one document from its text.

    Separated from file I/O so the parser is testable on strings, which is how
    the malformed-document tests avoid touching the real corpus.
    """
    where = source if source is not None else relative_path
    mapping, body = parse_front_matter(text, where)

    try:
        metadata = KnowledgeMetadata(**mapping)
    except ValidationError as exc:
        raise MetadataError(_format_validation_error(exc), where) from exc
    except TypeError as exc:  # non-string keys cannot occur, but be explicit
        raise MetadataError(str(exc), where) from exc

    sections = split_sections(body, where)

    return KnowledgeDocument(
        metadata=metadata,
        sections=sections,
        relative_path=relative_path,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _format_validation_error(exc: ValidationError) -> str:
    """Render a pydantic error compactly, one problem per line."""
    parts = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err["loc"]) or "<model>"
        parts.append(f"{location}: {err['msg']}")
    return "invalid front matter -- " + "; ".join(parts)


def load_document(path: Path, root: Path | None = None) -> KnowledgeDocument:
    """Load and validate a single document from disk."""
    base = root if root is not None else path.parent.parent
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise KnowledgeError(f"cannot read document: {exc}", path) from exc
    except UnicodeDecodeError as exc:
        raise KnowledgeError(f"document is not valid UTF-8: {exc}", path) from exc

    try:
        relative = path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        relative = path.name

    document = parse_document(text, relative, path)

    # The directory a document lives in must agree with the category it
    # declares; a misfiled document would otherwise pass every other check.
    declared = document.metadata.category
    parent = path.parent.name
    if parent in CATEGORY_DIRECTORIES and CATEGORY_DIRECTORIES[parent] is not declared:
        raise MetadataError(
            f"category is {declared.value!r} but the file is in {parent!r}/", path
        )

    # Filename and id must match, so a citation can be traced to a file by eye.
    if path.stem != document.metadata.id:
        raise MetadataError(
            f"id is {document.metadata.id!r} but the filename is {path.stem!r}.md", path
        )

    return document


def discover_documents(root: Path | None = None) -> tuple[Path, ...]:
    """Return every ``.md`` document under the category directories, sorted.

    ``MANIFEST.md`` and anything outside the six category directories is
    ignored, so corpus documentation can live beside the corpus.  Sorting is by
    POSIX relative path, which makes discovery order — and therefore the order
    problems are reported in — identical on every platform.
    """
    base = root if root is not None else default_knowledge_root()
    if not base.is_dir():
        raise KnowledgeError("knowledge root does not exist or is not a directory", base)

    found: list[Path] = []
    for directory in sorted(CATEGORY_DIRECTORIES):
        category_dir = base / directory
        if not category_dir.is_dir():
            continue
        found.extend(p for p in category_dir.glob("*.md") if p.is_file())

    return tuple(sorted(found, key=lambda p: p.relative_to(base).as_posix()))


def load_corpus(root: Path | None = None) -> tuple[KnowledgeDocument, ...]:
    """Load, validate and order the whole corpus.

    Ordering is ``(category declaration order, document id)`` — deterministic,
    stable across platforms, and independent of filesystem iteration order.
    Category order runs from reference material to adversary behaviour, so
    anything that consumes the corpus in order reads context before threats.

    Raises the first :class:`KnowledgeError` encountered.  A corpus loads
    completely or not at all; there is no partial success.
    """
    base = root if root is not None else default_knowledge_root()
    documents = [load_document(path, base) for path in discover_documents(base)]

    seen: dict[str, str] = {}
    for document in documents:
        if document.id in seen:
            raise DuplicateIdError(
                f"id {document.id!r} is already used by {seen[document.id]}",
                document.relative_path,
            )
        seen[document.id] = document.relative_path

    category_rank = {category: i for i, category in enumerate(Category)}
    documents.sort(key=lambda d: (category_rank[d.category], d.id))
    return tuple(documents)


# ===========================================================================
# Manual check:  python -m ai.rag.documents
# ===========================================================================
if __name__ == "__main__":  # pragma: no cover - manual check
    import sys

    try:
        corpus = load_corpus()
    except KnowledgeError as error:
        print(f"corpus failed to load\n  {error}")
        sys.exit(1)

    root = default_knowledge_root()
    print(f"corpus root: {root}")
    print(f"{len(corpus)} document(s), {sum(d.word_count() for d in corpus)} words\n")
    for doc in corpus:
        mitre = f" [{', '.join(doc.metadata.mitre)}]" if doc.metadata.mitre else ""
        print(f"  {doc.category.value:<22} {doc.id:<38} {doc.title}{mitre}")
        print(f"  {'':<22} signals: {', '.join(doc.metadata.applies_to)}")
        print(f"  {'':<22} licence: {doc.metadata.licence}  sha256: {doc.sha256[:12]}")
    print(f"\nall {len(REQUIRED_SECTIONS)} required sections present in every document")
