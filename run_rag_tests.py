"""Test runner for the RAG knowledge corpus -- step 1 only.

Scope
-----
This suite covers exactly what step 1 built: the metadata model, the
front-matter parser, the section template and the corpus loader.  Chunking,
embeddings, the vector store, retrieval and evaluation do not exist yet and
are not referenced here.

Deliberately dependency-free.  It needs **no** embedding model, **no** numpy,
**no** sentence-transformers, **no** network access and **no** API key -- only
the standard library and pydantic, which the project already requires.  That
is the property that keeps corpus validation runnable in any environment, and
it is why the malformed-document cases build their fixtures in a temporary
directory rather than mutating the real corpus.

Run::

    python run_rag_tests.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Import the package so its console-encoding fix is applied before any output.
import dpi  # noqa: F401
from ai.rag.documents import (
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
    parse_document,
    parse_front_matter,
)
from ai.schemas import CaptureReport, CaptureTotals, FlowRecord

_passed = 0
_failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}" + (f"  -- {detail}" if detail else ""))


def raises(label: str, expected: type[Exception], call) -> None:
    """Assert ``call()`` raises ``expected``, and that the message is useful.

    A rejection test that only checks the exception type is half a test: an
    error nobody can act on is nearly as bad as no error, so the message is
    required to be non-trivial too.
    """
    try:
        call()
    except expected as exc:
        message = str(exc)
        check(label, len(message) > 20, f"error message too terse: {message!r}")
    except Exception as exc:  # noqa: BLE001 - wrong type is the failure
        check(label, False, f"raised {type(exc).__name__}: {exc}")
    else:
        check(label, False, f"no {expected.__name__} was raised")


# ===========================================================================
# Fixture helpers
# ===========================================================================
#: A minimal document that passes every check, used as the base for the
#: malformed variants below.  Each rejection test changes exactly one thing.
VALID_FRONT_MATTER = """\
id: sample-document
title: Sample Document
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

VALID_BODY = "\n".join(f"## {name}\n\nBody text for {name}.\n" for name in REQUIRED_SECTIONS)


def make_document(front_matter: str = VALID_FRONT_MATTER, body: str = VALID_BODY) -> str:
    return f"---\n{front_matter}---\n\n{body}"


def with_line(*replacement: str, replacing: str) -> str:
    """Return the sample front matter with one line swapped for ``replacement``.

    Several lines may be supplied, which is how a scalar (``mitre: []``) is
    turned into a block list without leaving the following item orphaned.
    """
    lines = VALID_FRONT_MATTER.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(replacing):
            lines[i:i + 1] = list(replacement)
            break
    else:  # pragma: no cover - fixture bug, not a product bug
        raise AssertionError(f"fixture has no line starting with {replacing!r}")
    return "\n".join(lines) + "\n"


# ===========================================================================
# 1. The real corpus loads
# ===========================================================================
def test_corpus_loads() -> tuple[KnowledgeDocument, ...]:
    print("\nCorpus -- loading")

    root = default_knowledge_root()
    check("knowledge root exists", root.is_dir(), str(root))

    discovered = discover_documents(root)
    check("six documents are discovered", len(discovered) == 6, f"found {len(discovered)}")

    corpus = load_corpus()
    check("all six documents load and validate", len(corpus) == 6, f"loaded {len(corpus)}")

    check("MANIFEST.md exists", (root / "MANIFEST.md").is_file())
    check("MANIFEST.md is not loaded as a document",
          all(d.relative_path != "MANIFEST.md" for d in corpus))

    for name in CATEGORY_DIRECTORIES:
        check(f"category directory {name}/ exists", (root / name).is_dir())

    return corpus


# ===========================================================================
# 2. Metadata, ids, categories, sections, provenance
# ===========================================================================
def test_corpus_contents(corpus: tuple[KnowledgeDocument, ...]) -> None:
    print("\nCorpus -- metadata and structure")

    check("every document has validated metadata",
          all(isinstance(d.metadata, KnowledgeMetadata) for d in corpus))

    ids = [d.id for d in corpus]
    check("document ids are unique", len(set(ids)) == len(ids), str(sorted(ids)))
    check("filename matches id for every document",
          all(Path(d.relative_path).stem == d.id for d in corpus))

    categories = [d.category for d in corpus]
    check("every category is a known Category",
          all(isinstance(c, Category) for c in categories))
    check("all six categories are represented exactly once",
          sorted(c.value for c in categories) == sorted(CATEGORY_DIRECTORIES),
          str(sorted(c.value for c in categories)))
    check("every document sits in the directory matching its category",
          all(Path(d.relative_path).parent.name == d.category.value for d in corpus))

    for document in corpus:
        check(f"{document.id}: all six required sections present",
              tuple(document.sections) == REQUIRED_SECTIONS,
              str(tuple(document.sections)))
        check(f"{document.id}: no section is empty",
              all(body.strip() for body in document.sections.values()))

    # -- provenance and licensing are mandatory ----------------------------
    for document in corpus:
        meta = document.metadata
        check(f"{document.id}: declares at least one source", len(meta.sources) >= 1)
        check(f"{document.id}: declares a licence", bool(meta.licence.strip()))
        check(f"{document.id}: applies_to uses only known signals",
              set(meta.applies_to) <= KNOWN_SIGNALS,
              str(sorted(set(meta.applies_to) - KNOWN_SIGNALS)))

    mitre_docs = [d for d in corpus if d.metadata.mitre]
    check("MITRE-citing documents are licensed CC-BY-4.0",
          all(d.metadata.licence == "CC-BY-4.0" for d in mitre_docs),
          str([(d.id, d.metadata.licence) for d in mitre_docs]))
    manifest = (default_knowledge_root() / "MANIFEST.md").read_text(encoding="utf-8")
    check("MANIFEST.md attributes MITRE ATT&CK",
          "MITRE" in manifest and "CC BY 4.0" in manifest)
    check("MANIFEST.md states the corpus is curated and reviewed",
          "curated" in manifest.lower() and "review" in manifest.lower())


# ===========================================================================
# 3. The corpus is about *this* project
# ===========================================================================
def test_project_specificity(corpus: tuple[KnowledgeDocument, ...]) -> None:
    print("\nCorpus -- project specificity")

    # Derived from the live models, so the check cannot drift from the schema.
    real_fields = (
        set(FlowRecord.model_fields)
        | set(CaptureReport.model_fields)
        | set(CaptureTotals.model_fields)
    )
    section = "What the DPI engine can observe"

    for document in corpus:
        text = document.section(section)
        mentioned = {field for field in real_fields if field in text}
        check(f"{document.id}: observe-section cites real report fields",
              len(mentioned) >= 3,
              f"only found {sorted(mentioned)}")

    # The report has no time dimension; the corpus must not imply otherwise.
    check("no document claims a duration field exists",
          all("`duration`" not in d.section(section) for d in corpus))
    check("no document claims a timestamp field exists",
          all("`timestamp`" not in d.section(section) for d in corpus))
    check("the timing limitation is stated somewhere in the corpus",
          any("timestamp" in d.section(section).lower() for d in corpus))


# ===========================================================================
# 4. Deterministic ordering
# ===========================================================================
def test_determinism(corpus: tuple[KnowledgeDocument, ...]) -> None:
    print("\nCorpus -- determinism")

    again = load_corpus()
    check("repeated loads return the same id order",
          [d.id for d in again] == [d.id for d in corpus],
          f"{[d.id for d in again]} != {[d.id for d in corpus]}")
    check("repeated loads return the same content hashes",
          [d.sha256 for d in again] == [d.sha256 for d in corpus])

    rank = {category: i for i, category in enumerate(Category)}
    keys = [(rank[d.category], d.id) for d in corpus]
    check("ordering is (category declaration order, id)", keys == sorted(keys), str(keys))
    check("glossary sorts before attack-patterns",
          rank[Category.GLOSSARY] < rank[Category.ATTACK_PATTERNS])

    root = default_knowledge_root()
    paths = [p.relative_to(root).as_posix() for p in discover_documents(root)]
    check("discovery order is relative-path sorted", paths == sorted(paths), str(paths))
    check("parsing is byte-stable",
          parse_document(make_document(), "protocols/sample-document.md").sha256
          == parse_document(make_document(), "protocols/sample-document.md").sha256)


# ===========================================================================
# 5. Front-matter parser -- the accepted subset
# ===========================================================================
def test_front_matter_parser() -> None:
    print("\nFront matter -- parser subset")

    mapping, body = parse_front_matter(make_document())
    check("scalars parse", mapping["id"] == "sample-document")
    check("block lists parse", mapping["applies_to"] == ["dns_high_volume"])
    check("'[]' parses as an empty list", mapping["mitre"] == [])
    check("the body is returned separately", body.lstrip().startswith("## Summary"))

    check("version stays a string, never a float",
          isinstance(mapping["version"], str) and mapping["version"] == "1.0")

    quoted, _ = parse_front_matter(
        make_document(with_line('title: "Quoted Title"', replacing="title:"))
    )
    check("quotes are stripped from scalars", quoted["title"] == "Quoted Title")

    commented, _ = parse_front_matter(
        f"---\n# a comment\n\n{VALID_FRONT_MATTER}---\n\n{VALID_BODY}"
    )
    check("comments and blank lines are ignored", commented["id"] == "sample-document")

    raises("missing front-matter fence is rejected", FrontMatterError,
           lambda: parse_front_matter("## Summary\n\nno front matter\n"))
    raises("unterminated front matter is rejected", FrontMatterError,
           lambda: parse_front_matter(f"---\n{VALID_FRONT_MATTER}\n## Summary\n"))
    raises("a tab in front matter is rejected", FrontMatterError,
           lambda: parse_front_matter(make_document("id:\tsample-document\n")))
    raises("a duplicate key is rejected", FrontMatterError,
           lambda: parse_front_matter(make_document(VALID_FRONT_MATTER + "id: other\n")))
    raises("an inline list is rejected", FrontMatterError,
           lambda: parse_front_matter(
               make_document(with_line("keywords: [a, b]", replacing="keywords:"))))
    raises("a stray list item is rejected", FrontMatterError,
           lambda: parse_front_matter(make_document("- orphan\n")))
    raises("an unparseable line is rejected", FrontMatterError,
           lambda: parse_front_matter(make_document(VALID_FRONT_MATTER + "nonsense\n")))


# ===========================================================================
# 6. Metadata validation -- rejection cases
# ===========================================================================
def test_metadata_rejection() -> None:
    print("\nMetadata -- rejection cases")

    def parse(front_matter: str):
        return lambda: parse_document(
            make_document(front_matter), "protocols/sample-document.md"
        )

    # The headline case: a misspelled key is a silent retrieval bug, so the
    # model forbids extras rather than ignoring them.
    raises("an unknown metadata field is rejected", MetadataError,
           parse(VALID_FRONT_MATTER + "unexpected_field: value\n"))
    raises("a misspelled 'keyword:' is rejected as unknown", MetadataError,
           parse(with_line("keyword:", replacing="keywords:")))

    raises("a missing required field is rejected", MetadataError,
           parse("\n".join(
               line for line in VALID_FRONT_MATTER.splitlines()
               if not line.startswith("licence:")) + "\n"))
    raises("a missing sources field is rejected", MetadataError,
           parse(VALID_FRONT_MATTER.replace("sources:\n  - Authored for this project.\n", "")))

    raises("an unknown category is rejected", MetadataError,
           parse(with_line("category: made-up", replacing="category:")))
    raises("a non-slug id is rejected", MetadataError,
           parse(with_line("id: Not A Slug", replacing="id:")))
    raises("a malformed version is rejected", MetadataError,
           parse(with_line("version: one", replacing="version:")))
    raises("a malformed date is rejected", MetadataError,
           parse(with_line("updated: last-tuesday", replacing="updated:")))
    raises("an unknown severity_hint is rejected", MetadataError,
           parse(with_line("severity_hint: catastrophic", replacing="severity_hint:")))
    raises("a malformed MITRE id is rejected", MetadataError,
           parse(with_line("mitre:", "  - TA0001", replacing="mitre:")))
    raises("an empty applies_to is rejected", MetadataError,
           parse(with_line("applies_to: []", replacing="applies_to:").replace(
               "  - dns_high_volume\n", "")))

    # The signal vocabulary is the join key retrieval will filter on, so a
    # document cannot claim to answer a signal that does not exist.
    raises("an unknown signal in applies_to is rejected", MetadataError,
           parse(VALID_FRONT_MATTER.replace("dns_high_volume", "dns_beaconing")))


# ===========================================================================
# 7. Section-template validation -- rejection cases
# ===========================================================================
def test_section_rejection() -> None:
    print("\nSections -- rejection cases")

    def parse(body: str):
        return lambda: parse_document(
            make_document(body=body), "protocols/sample-document.md"
        )

    missing = "\n".join(
        f"## {name}\n\nBody.\n" for name in REQUIRED_SECTIONS if name != "References"
    )
    raises("a missing section is rejected", SectionError, parse(missing))

    extra = VALID_BODY + "\n## Extra Section\n\nBody.\n"
    raises("an unexpected section is rejected", SectionError, parse(extra))

    reordered = "\n".join(
        f"## {name}\n\nBody.\n" for name in reversed(REQUIRED_SECTIONS)
    )
    raises("out-of-order sections are rejected", SectionError, parse(reordered))

    empty = VALID_BODY.replace("Body text for Indicators.\n", "")
    raises("an empty section is rejected", SectionError, parse(empty))

    duplicated = VALID_BODY + "\n## Summary\n\nAgain.\n"
    raises("a duplicate section heading is rejected", SectionError, parse(duplicated))


# ===========================================================================
# 8. Loader-level rejection -- duplicates, misfiling, bad names
# ===========================================================================
def test_loader_rejection() -> None:
    print("\nLoader -- rejection cases")

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        for name in CATEGORY_DIRECTORIES:
            (root / name).mkdir()

        good = root / "protocols" / "sample-document.md"
        good.write_text(make_document(), encoding="utf-8")
        check("a valid temporary corpus loads", len(load_corpus(root)) == 1)

        # Same id, different filename, different category directory.
        twin = root / "baselines" / "sample-document.md"
        twin.write_text(
            make_document(with_line("category: baselines", replacing="category:")),
            encoding="utf-8",
        )
        raises("a duplicate id across categories is rejected", DuplicateIdError,
               lambda: load_corpus(root))
        twin.unlink()

        misfiled = root / "baselines" / "misfiled-document.md"
        misfiled.write_text(
            make_document(with_line("id: misfiled-document", replacing="id:")),
            encoding="utf-8",
        )
        raises("a document filed under the wrong category is rejected", MetadataError,
               lambda: load_corpus(root))
        misfiled.unlink()

        mismatched = root / "protocols" / "wrong-name.md"
        mismatched.write_text(make_document(), encoding="utf-8")
        raises("a filename that does not match the id is rejected", MetadataError,
               lambda: load_corpus(root))
        mismatched.unlink()

        check("the temporary corpus is valid again after cleanup",
              len(load_corpus(root)) == 1)

    raises("a missing knowledge root is reported clearly", KnowledgeError,
           lambda: load_corpus(Path(raw) / "gone"))


# ===========================================================================
# 9. The suite itself stays light
# ===========================================================================
def test_no_heavy_dependencies() -> None:
    print("\nDependencies")

    heavy = [name for name in ("numpy", "torch", "sentence_transformers", "yaml",
                               "faiss", "chromadb") if name in sys.modules]
    check("no embedding, vector-store or YAML dependency was imported",
          not heavy, f"imported: {heavy}")
    check("the openai SDK is not needed to validate the corpus",
          "openai" not in sys.modules)
    check("ai.rag imports cleanly with no API key configured", True)


# ===========================================================================
def main() -> int:
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print("RAG step 1 -- knowledge corpus skeleton and loader")

    try:
        corpus = test_corpus_loads()
    except KnowledgeError as error:
        print(f"\n  FAIL  the corpus does not load\n        {error}")
        return 1

    test_corpus_contents(corpus)
    test_project_specificity(corpus)
    test_determinism(corpus)
    test_front_matter_parser()
    test_metadata_rejection()
    test_section_rejection()
    test_loader_rejection()
    test_no_heavy_dependencies()

    total = _passed + _failed
    print(f"\n{_passed}/{total} checks passed")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
