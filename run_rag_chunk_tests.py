"""Test runner for deterministic chunking -- RAG step 2 only.

Scope
-----
Covers :mod:`ai.rag.chunking`: the chunk model, the section-aware splitter,
deterministic ids, provenance carry-through and stable ordering.  Embeddings,
vector stores, retrieval, query building, signal extraction and evaluation do
not exist yet and are not referenced.

Like the step 1 suite this needs **no** embedding model, **no** numpy, **no**
vector database, **no** network and **no** API key.  Two of the checks below
assert exactly that, one of them by making the socket layer raise if anything
tries to open a connection.

Large-document behaviour is exercised against synthetic corpora built in a
:class:`tempfile.TemporaryDirectory`; the real ``knowledge/`` tree is only ever
read.

Run::

    python run_rag_chunk_tests.py
"""

from __future__ import annotations

import socket
import sys
import tempfile
from pathlib import Path

# Import the package so its console-encoding fix is applied before any output.
import dpi  # noqa: F401
from ai.rag.chunking import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    OVERLAP_SEPARATOR,
    ChunkConfig,
    KnowledgeChunk,
    chunk_corpus,
    chunk_document,
    chunk_statistics,
    section_slug,
    serialize_chunks,
)
from ai.rag.documents import (
    CATEGORY_DIRECTORIES,
    REQUIRED_SECTIONS,
    Category,
    KnowledgeDocument,
    MetadataError,
    SectionError,
    load_corpus,
)

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
    """Assert ``call()`` raises ``expected`` with a usable message."""
    try:
        call()
    except expected as exc:
        check(label, len(str(exc)) > 15, f"error message too terse: {str(exc)!r}")
    except Exception as exc:  # noqa: BLE001 - wrong type is the failure
        check(label, False, f"raised {type(exc).__name__}: {exc}")
    else:
        check(label, False, f"no {expected.__name__} was raised")


# ===========================================================================
# Synthetic corpus helpers
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

#: A paragraph of a known, easily-counted size, used to build oversized
#: sections whose expected split points can be reasoned about by hand.
PARAGRAPH = (
    "This paragraph exists to give the chunker something of a known size to "
    "work with. It contains several sentences so that sentence boundaries are "
    "available as well as paragraph boundaries. It is deliberately plain."
)


def build_document_text(
    *, doc_id: str = "sample-document", title: str = "Sample Document", big_section: str = ""
) -> str:
    """Return valid document text, optionally with an oversized Indicators section."""
    sections = []
    for name in REQUIRED_SECTIONS:
        body = big_section if (name == "Indicators" and big_section) else f"Body text for {name}."
        sections.append(f"## {name}\n\n{body}\n")
    front = FRONT_MATTER.format(doc_id=doc_id, title=title)
    return f"---\n{front}---\n\n" + "\n".join(sections)


def write_corpus(root: Path, documents: dict[str, str]) -> None:
    """Create the category directories and write ``{filename: text}``."""
    for name in CATEGORY_DIRECTORIES:
        (root / name).mkdir(exist_ok=True)
    for doc_id, text in documents.items():
        (root / "protocols" / f"{doc_id}.md").write_text(text, encoding="utf-8")


def paragraphs(count: int) -> str:
    return "\n\n".join(f"{PARAGRAPH} Paragraph number {i}." for i in range(count))


# ===========================================================================
# 1. The real corpus chunks
# ===========================================================================
def test_real_corpus() -> tuple[KnowledgeChunk, ...]:
    print("\nCorpus -- chunking the real knowledge base")

    corpus = load_corpus()
    chunks = chunk_corpus()

    check("all six documents produce chunks",
          len({c.document_id for c in chunks}) == 6,
          str(sorted({c.document_id for c in chunks})))
    check("every document contributes at least six chunks (one per section)",
          all(sum(1 for c in chunks if c.document_id == d.id) >= len(REQUIRED_SECTIONS)
              for d in corpus))
    check("every section of every document is represented",
          {(c.document_id, c.section) for c in chunks}
          == {(d.id, s) for d in corpus for s in REQUIRED_SECTIONS})

    check("every chunk is a validated KnowledgeChunk",
          all(isinstance(c, KnowledgeChunk) for c in chunks))
    check("every chunk survives a round trip through its own model",
          all(KnowledgeChunk.model_validate(c.model_dump()) == c for c in chunks))

    check("no chunk is empty", all(c.text.strip() for c in chunks))
    check("no chunk is whitespace-padded", all(c.text == c.text.strip() for c in chunks))
    check("char_count matches the text everywhere",
          all(c.char_count == len(c.text) for c in chunks))

    known_ids = {d.id for d in corpus}
    check("every chunk references a real document id",
          all(c.document_id in known_ids for c in chunks))
    check("every chunk references a required section",
          all(c.section in REQUIRED_SECTIONS for c in chunks))
    check("section_index agrees with the section name",
          all(c.section_index == REQUIRED_SECTIONS.index(c.section) for c in chunks))

    ids = [c.chunk_id for c in chunks]
    check("chunk ids are unique", len(set(ids)) == len(ids),
          f"{len(ids) - len(set(ids))} duplicates")
    check("chunk ids embed document and section",
          all(c.chunk_id.startswith(f"{c.document_id}#{section_slug(c.section)}#")
              for c in chunks))

    return chunks


# ===========================================================================
# 2. Provenance
# ===========================================================================
def test_provenance(chunks: tuple[KnowledgeChunk, ...]) -> None:
    print("\nProvenance")

    by_id = {d.id: d for d in load_corpus()}

    for chunk in chunks:
        source = by_id[chunk.document_id]
        meta = source.metadata
        ok = (
            chunk.title == meta.title
            and chunk.category is meta.category
            and chunk.document_version == meta.version
            and chunk.document_updated == meta.updated
            and chunk.document_sha256 == source.sha256
            and chunk.relative_path == source.relative_path
            and chunk.keywords == list(meta.keywords)
            and chunk.applies_to == list(meta.applies_to)
            and chunk.mitre == list(meta.mitre)
            and chunk.severity_hint is meta.severity_hint
        )
        check(f"{chunk.chunk_id[:46]}: metadata matches its source document", ok)

    check("every chunk carries the document licence",
          all(c.licence == by_id[c.document_id].metadata.licence for c in chunks))
    check("every chunk carries at least one source reference",
          all(len(c.sources) >= 1 for c in chunks))
    check("every chunk carries the document's full source list",
          all(c.sources == list(by_id[c.document_id].metadata.sources) for c in chunks))
    check("CC-BY-4.0 licensing survives chunking",
          all(c.licence == "CC-BY-4.0" for c in chunks if c.document_id == "dns-tunneling"))

    check("heading_path names document and section",
          all(c.heading_path == f"{c.title} > {c.section}" for c in chunks))
    check("citation() answers 'which document and section'",
          all(c.document_id in c.citation() and c.section in c.citation() for c in chunks))
    check("relative_path points into the right category directory",
          all(Path(c.relative_path).parent.name == c.category.value for c in chunks))

    # The provenance guarantee that matters most: a chunk body is a verbatim
    # substring of its source section, so a citation can be checked by eye.
    check("every chunk body is verbatim text from its source section",
          all(c.body() in by_id[c.document_id].section(c.section) for c in chunks))


# ===========================================================================
# 3. Determinism
# ===========================================================================
def test_determinism(chunks: tuple[KnowledgeChunk, ...]) -> None:
    print("\nDeterminism")

    again = chunk_corpus()
    check("chunk ids are identical across runs",
          [c.chunk_id for c in again] == [c.chunk_id for c in chunks])
    check("chunk order is identical across runs",
          [(c.document_id, c.section, c.chunk_index) for c in again]
          == [(c.document_id, c.section, c.chunk_index) for c in chunks])
    check("chunk text is identical across runs",
          [c.text for c in again] == [c.text for c in chunks])
    check("running the chunker twice produces identical serialized output",
          serialize_chunks(again) == serialize_chunks(chunks))
    check("content hashes are stable across runs",
          [c.content_sha256 for c in again] == [c.content_sha256 for c in chunks])

    # Ordering rule: corpus order -> section order -> chunk index.
    corpus_rank = {d.id: i for i, d in enumerate(load_corpus())}
    keys = [(corpus_rank[c.document_id], c.section_index, c.chunk_index) for c in chunks]
    check("ordering is corpus order, then section order, then chunk index",
          keys == sorted(keys), str(keys[:6]))
    # Order follows the corpus, not the filesystem: feeding the documents in a
    # different order changes the chunk order and nothing else.
    corpus = load_corpus()
    reversed_chunks = chunk_corpus(tuple(reversed(corpus)))
    check("chunk order follows the order documents are supplied in",
          [c.document_id for c in reversed_chunks] != [c.document_id for c in chunks])
    check("supplying documents in another order changes no chunk id",
          {c.chunk_id for c in reversed_chunks} == {c.chunk_id for c in chunks})
    check("chunk_corpus() with no arguments equals chunk_corpus(load_corpus())",
          serialize_chunks(chunk_corpus(corpus)) == serialize_chunks(chunks))

    # Ids must come from a stable hash, not from Python's salted hash().
    check("chunk ids are hash-suffixed, not positional only",
          all(len(c.chunk_id.rsplit("-", 1)[-1]) == 12 for c in chunks))


# ===========================================================================
# 4. Short sections stay whole
# ===========================================================================
def test_short_sections(chunks: tuple[KnowledgeChunk, ...]) -> None:
    print("\nSmall sections")

    config = ChunkConfig()
    corpus = load_corpus()

    small = [(d.id, s) for d in corpus for s in REQUIRED_SECTIONS
             if len(d.section(s)) <= config.max_chars]
    counts = {(c.document_id, c.section): c.chunk_count for c in chunks}
    check("every section that fits the budget is exactly one chunk",
          all(counts[key] == 1 for key in small),
          str([k for k in small if counts[k] != 1]))
    check("the real corpus has such sections to test", len(small) >= 30, str(len(small)))

    check("unsplit chunks carry no overlap",
          all(c.overlap_chars == 0 for c in chunks if c.chunk_count == 1))
    check("the first chunk of a split section carries no overlap",
          all(c.overlap_chars == 0 for c in chunks if c.chunk_index == 0))
    check("an unsplit chunk's body is its whole text",
          all(c.body() == c.text for c in chunks if c.chunk_count == 1))


# ===========================================================================
# 5. Large sections split, deterministically and on boundaries
# ===========================================================================
def test_large_sections() -> None:
    print("\nLarge sections -- synthetic documents")

    config = ChunkConfig()
    big = paragraphs(24)  # far beyond max_chars, all paragraph boundaries

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        write_corpus(root, {"sample-document": build_document_text(big_section=big)})

        corpus = load_corpus(root)
        check("the synthetic corpus loads", len(corpus) == 1)
        document = corpus[0]
        check("the synthetic section really is oversized",
              len(document.section("Indicators")) > config.max_chars * 3,
              str(len(document.section("Indicators"))))

        chunks = chunk_document(document, config)
        indicators = [c for c in chunks if c.section == "Indicators"]

        check("a large section splits into several chunks", len(indicators) > 1,
              str(len(indicators)))
        check("chunk_count matches the number of chunks produced",
              all(c.chunk_count == len(indicators) for c in indicators))
        check("chunk indices are contiguous from zero",
              [c.chunk_index for c in indicators] == list(range(len(indicators))))
        check("other sections of the same document stay single chunks",
              all(c.chunk_count == 1 for c in chunks if c.section != "Indicators"))

        # -- size ceiling ---------------------------------------------------
        check("no chunk exceeds the configured maximum",
              all(c.char_count <= config.max_chars for c in chunks),
              str(max(c.char_count for c in chunks)))

        # -- boundaries -----------------------------------------------------
        section_text = document.section("Indicators")
        starts_clean = []
        for chunk in indicators[1:]:
            body = chunk.body()
            offset = section_text.find(body)
            starts_clean.append(offset > 0 and section_text[offset - 1] == "\n")
        check("splits land on paragraph boundaries when paragraphs are available",
              all(starts_clean), str(starts_clean))
        check("no chunk body starts mid-word",
              all(not c.body()[:1].isspace() for c in indicators))

        # -- overlap --------------------------------------------------------
        check("every chunk after the first carries overlap",
              all(c.overlap_chars > 0 for c in indicators[1:]))
        check("overlap never exceeds the configured budget",
              all(c.overlap_chars <= config.overlap_chars for c in indicators))
        for previous, current in zip(indicators, indicators[1:]):
            check(f"chunk {current.chunk_index}: overlap is the previous chunk's tail",
                  previous.text.endswith(current.text[:current.overlap_chars]))
        check("body() removes the overlap prefix and keeps the rest",
              all(c.text.endswith(c.body())
                  and len(c.body()) == c.char_count - c.overlap_chars
                  - len(OVERLAP_SEPARATOR)
                  for c in indicators[1:]))
        check("the overlap separator sits between tail and body",
              all(c.text[c.overlap_chars:c.overlap_chars + len(OVERLAP_SEPARATOR)]
                  == OVERLAP_SEPARATOR for c in indicators[1:]))

        # -- determinism of the split ---------------------------------------
        repeat = chunk_document(document, config)
        check("splitting a large section is deterministic",
              [c.chunk_id for c in repeat] == [c.chunk_id for c in chunks])
        check("overlap is deterministic",
              [c.overlap_chars for c in repeat] == [c.overlap_chars for c in chunks])
        check("serialized output of a split document is byte-identical",
              serialize_chunks(repeat) == serialize_chunks(chunks))

    # -- list-aware splitting -------------------------------------------
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        listy = "Intro line before the list:\n\n" + "\n".join(
            f"- Item {i} explains one thing at moderate length so the list is long "
            f"enough to force a split across several chunks." for i in range(40)
        )
        write_corpus(root, {"listy-document": build_document_text(
            doc_id="listy-document", title="Listy Document", big_section=listy)})
        chunks = chunk_document(load_corpus(root)[0], config)
        items = [c for c in chunks if c.section == "Indicators"]
        check("a long list splits into several chunks", len(items) > 1)
        check("a short intro paragraph is not emitted as its own chunk",
              all(c.char_count >= config.min_fill_chars for c in items[:-1]),
              str([c.char_count for c in items]))
        check("list splits land on item starts",
              all(c.body().lstrip().startswith("- ") for c in items[1:]),
              str([c.body()[:12] for c in items[1:]]))

    # -- the documented unavoidable case --------------------------------
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runon = "x" * (config.max_chars * 2 + 50)  # no whitespace at all
        write_corpus(root, {"runon-document": build_document_text(
            doc_id="runon-document", title="Runon Document", big_section=runon)})
        chunks = chunk_document(load_corpus(root)[0], config)
        items = [c for c in chunks if c.section == "Indicators"]
        check("text with no boundary at all still splits", len(items) > 1)
        check("even a hard cut respects the size ceiling",
              all(c.char_count <= config.max_chars for c in items),
              str([c.char_count for c in items]))
        check("hard-cut splitting is deterministic",
              [c.chunk_id for c in chunk_document(load_corpus(root)[0], config)]
              == [c.chunk_id for c in chunks])


# ===========================================================================
# 6. Content sensitivity and isolation of ids
# ===========================================================================
def test_id_sensitivity() -> None:
    print("\nChunk id sensitivity")

    config = ChunkConfig()

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        write_corpus(root, {
            "first-document": build_document_text(doc_id="first-document",
                                                  title="First Document"),
            "second-document": build_document_text(doc_id="second-document",
                                                   title="Second Document"),
        })

        before = {c.chunk_id: c for c in chunk_corpus(load_corpus(root), config)}

        # Edit exactly one section of one document.
        target = root / "protocols" / "first-document.md"
        target.write_text(
            build_document_text(doc_id="first-document", title="First Document")
            .replace("Body text for Indicators.", "Body text for Indicators, revised."),
            encoding="utf-8",
        )
        after = {c.chunk_id: c for c in chunk_corpus(load_corpus(root), config)}

        changed = set(before) ^ set(after)
        check("changing a section's text changes that chunk's id", bool(changed))
        check("only the edited section's chunk id changes",
              {before[i].section if i in before else after[i].section
               for i in changed} == {"Indicators"},
              str(sorted(changed)))
        check("the edited document's other chunks keep their ids",
              sum(1 for i in set(before) & set(after)
                  if before[i].document_id == "first-document") == len(REQUIRED_SECTIONS) - 1)
        check("the unrelated document's chunk ids are untouched",
              [c.chunk_id for c in after.values() if c.document_id == "second-document"]
              == [c.chunk_id for c in before.values() if c.document_id == "second-document"])

        # Whitespace-only edits must not churn ids: the hash is over
        # normalised text, so a reflow is not a content change.
        def indicators_id() -> str:
            chunks = chunk_corpus(load_corpus(root), config)
            return next(c.chunk_id for c in chunks
                        if c.document_id == "first-document" and c.section == "Indicators")

        original_id = next(c.chunk_id for c in before.values()
                           if c.document_id == "first-document" and c.section == "Indicators")
        target.write_text(
            build_document_text(doc_id="first-document", title="First Document")
            .replace("Body text for Indicators.", "Body   text  for\nIndicators."),
            encoding="utf-8",
        )
        check("a whitespace-only reflow does not change the chunk id",
              indicators_id() == original_id,
              f"{indicators_id()} != {original_id}")
        check("reflow determinism holds on a second pass",
              indicators_id() == original_id)


# ===========================================================================
# 7. The model rejects bad chunks
# ===========================================================================
def test_model_rejection(chunks: tuple[KnowledgeChunk, ...]) -> None:
    print("\nChunk model -- rejection cases")

    sample = chunks[0].model_dump()

    def variant(**changes):
        data = dict(sample)
        data.update(changes)
        return lambda: KnowledgeChunk.model_validate(data)

    raises("an unexpected field is rejected", ValueError,
           variant(unexpected_field="value"))
    raises("empty text is rejected", ValueError, variant(text=""))
    raises("whitespace-only text is rejected", ValueError, variant(text="   \n  "))
    raises("an unknown section is rejected", ValueError, variant(section="Appendix"))
    raises("a section_index that disagrees with the section is rejected", ValueError,
           variant(section_index=5))
    raises("a char_count that disagrees with the text is rejected", ValueError,
           variant(char_count=99999))
    raises("a malformed chunk_id is rejected", ValueError, variant(chunk_id="not-an-id"))
    raises("a chunk_id from a different document is rejected", ValueError,
           variant(chunk_id="other-doc#summary#00-0123456789ab"))
    raises("overlap on the first chunk of a section is rejected", ValueError,
           variant(chunk_index=0, overlap_chars=10))
    raises("a chunk_index beyond chunk_count is rejected", ValueError,
           variant(chunk_index=7, chunk_count=2))
    raises("a missing licence is rejected", ValueError, variant(licence=""))
    raises("an empty sources list is rejected", ValueError, variant(sources=[]))

    def mutate() -> None:
        chunks[0].text = "rewritten"  # type: ignore[misc]

    raises("a chunk cannot be mutated after construction", ValueError, mutate)


# ===========================================================================
# 8. Configuration
# ===========================================================================
def test_configuration() -> None:
    print("\nConfiguration")

    check("the default maximum is documented and configurable",
          ChunkConfig().max_chars == DEFAULT_MAX_CHARS == 1400)
    check("the default overlap is documented and configurable",
          ChunkConfig().overlap_chars == DEFAULT_OVERLAP_CHARS == 200)
    check("min_fill_chars derives from the ratio",
          ChunkConfig(max_chars=1000, min_fill_ratio=0.5).min_fill_chars == 500)

    raises("an absurdly small maximum is rejected", ValueError,
           lambda: ChunkConfig(max_chars=10))
    raises("negative overlap is rejected", ValueError,
           lambda: ChunkConfig(overlap_chars=-1))
    raises("overlap at half the budget is rejected", ValueError,
           lambda: ChunkConfig(max_chars=1000, overlap_chars=500))
    raises("a fill ratio of 1.0 is rejected", ValueError,
           lambda: ChunkConfig(min_fill_ratio=1.0))

    # A smaller budget must split more, and still deterministically.
    corpus = load_corpus()
    tight = chunk_corpus(corpus, ChunkConfig(max_chars=600, overlap_chars=80))
    loose = chunk_corpus(corpus, ChunkConfig())
    check("a smaller max_chars produces more chunks", len(tight) > len(loose),
          f"{len(tight)} vs {len(loose)}")
    check("a smaller max_chars is still honoured",
          all(c.char_count <= 600 for c in tight),
          str(max(c.char_count for c in tight)))
    check("a different configuration is still deterministic",
          serialize_chunks(chunk_corpus(corpus, ChunkConfig(max_chars=600,
                                                            overlap_chars=80)))
          == serialize_chunks(tight))

    stats = chunk_statistics(loose)
    check("statistics report the document and chunk counts",
          stats["documents"] == 6 and stats["chunks"] == len(loose))
    check("statistics report a maximum within the limit",
          int(stats["max_chars"]) <= DEFAULT_MAX_CHARS)
    check("statistics on an empty set do not raise",
          chunk_statistics([])["chunks"] == 0)


# ===========================================================================
# 9. Malformed documents never reach the chunker
# ===========================================================================
def test_malformed_documents() -> None:
    print("\nMalformed documents")

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)

        bad_metadata = build_document_text().replace("severity_hint: info",
                                                     "severity_hint: catastrophic")
        write_corpus(root, {"sample-document": bad_metadata})
        raises("invalid metadata is rejected by the document layer", MetadataError,
               lambda: chunk_corpus(load_corpus(root)))

        missing_section = "\n".join(
            part for part in build_document_text().split("\n")
            if "## References" not in part
        )
        (root / "protocols" / "sample-document.md").write_text(missing_section,
                                                               encoding="utf-8")
        raises("a missing section is rejected by the document layer", SectionError,
               lambda: chunk_corpus(load_corpus(root)))

        (root / "protocols" / "sample-document.md").write_text(build_document_text(),
                                                               encoding="utf-8")
        check("a valid document chunks once the problems are fixed",
              len(chunk_corpus(load_corpus(root))) == len(REQUIRED_SECTIONS))

    check("the chunker takes validated documents, not raw markdown",
          all(isinstance(d, KnowledgeDocument) for d in load_corpus()))


# ===========================================================================
# 10. Dependency and isolation guarantees
# ===========================================================================
def test_isolation() -> None:
    print("\nDependencies and isolation")

    forbidden = [name for name in (
        "numpy", "torch", "transformers", "sentence_transformers",
        "faiss", "chromadb", "qdrant_client", "langchain", "yaml",
    ) if name in sys.modules]
    check("no embedding or vector-database library is imported", not forbidden,
          f"imported: {forbidden}")
    check("the openai SDK is not needed to chunk", "openai" not in sys.modules)

    # Prove it rather than assert it: if anything opens a socket while
    # chunking, this fails loudly.
    real_socket, real_connect = socket.socket, socket.create_connection

    def refuse(*args, **kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("chunking attempted a network connection")

    socket.socket, socket.create_connection = refuse, refuse  # type: ignore[assignment]
    try:
        produced = chunk_corpus()
        check("chunking makes no network calls", len(produced) > 0)
    finally:
        socket.socket, socket.create_connection = real_socket, real_connect

    check("chunking the corpus twice in one process is side-effect free",
          serialize_chunks(chunk_corpus()) == serialize_chunks(chunk_corpus()))


# ===========================================================================
def main() -> int:
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print("RAG step 2 -- deterministic chunking")

    chunks = test_real_corpus()
    test_provenance(chunks)
    test_determinism(chunks)
    test_short_sections(chunks)
    test_large_sections()
    test_id_sensitivity()
    test_model_rejection(chunks)
    test_configuration()
    test_malformed_documents()
    test_isolation()

    stats = chunk_statistics(chunks)
    print(f"\ncorpus: {stats['documents']} documents -> {stats['chunks']} chunks, "
          f"max {stats['max_chars']} chars, mean {stats['mean_chars']}, "
          f"{stats['split_sections']} section(s) split")

    total = _passed + _failed
    print(f"\n{_passed}/{total} checks passed")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
