"""Test runner for signal-driven retrieval -- RAG step 6 only.

Two tiers
---------
**Unit tier** -- always runs.  Query construction, the bge prefix, merging,
deduplication, ranking, diversity, provenance, model-compatibility guards and
privacy.  Retrieval mechanics are exercised through the ``encoder`` seam that
:class:`~ai.rag.embeddings.EmbeddingModel` already provides, using a keyword
encoder defined *in this file*: its vectors are a fixed keyword axis, so which
chunk should win any given query can be worked out by hand.  Nothing in
``ai/rag`` can ever select it -- the library has no fallback encoder.

**Integration tier** -- runs only when ``BAAI/bge-small-en-v1.5`` loads.  It
builds the real index once and retrieves against it.  Otherwise those checks
SKIP with the reason and the suite still passes.

Query construction needs no model at all, so the whole of section A runs
everywhere.

Run::

    python run_rag_retrieval_tests.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sys
from pathlib import Path

# Import the package so its console-encoding fix is applied before any output.
import dpi  # noqa: F401
from ai.schemas import (
    CaptureReport,
    CaptureTotals,
    FlowRecord,
    Severity,
    TransportProtocol,
)
from ai.rag.chunking import KnowledgeChunk, chunk_corpus, chunk_document
from ai.rag.documents import load_corpus, parse_document
from ai.rag.embeddings import (
    DEFAULT_MODEL,
    EmbeddingConfig,
    EmbeddingModel,
    EmbeddingResult,
    ModelUnavailableError,
    sentence_transformers_available,
)
from ai.rag.retrieval import (
    BGE_QUERY_PREFIX,
    CAPTURE_QUERY_LABEL,
    QUERY_TEMPLATES,
    RETRIEVAL_SCHEMA_VERSION,
    ModelMismatchError,
    RetrievalConfig,
    RetrievalError,
    RetrievalReport,
    RetrievedChunk,
    SignalQuery,
    apply_query_prefix,
    build_queries,
    retrieve_for_signals,
)
from ai.rag.signals import Signal, SignalReport, SignalType, SourceField, extract_signals
from ai.rag.vector_store import VectorRecord, VectorStore

_passed = 0
_failed = 0
_skipped = 0

MODEL_NAME = "test/keyword-6d"
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


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
    try:
        call()
    except expected as exc:
        check(label, len(str(exc)) > 10, f"error message too terse: {str(exc)!r}")
    except Exception as exc:  # noqa: BLE001 - wrong type is the failure
        check(label, False, f"raised {type(exc).__name__}: {exc}")
    else:
        check(label, False, f"no {expected.__name__} was raised")


# ===========================================================================
# Test doubles and fixtures
# ===========================================================================
class KeywordEncoder:
    """Six keyword axes, so every retrieval outcome is predictable by hand.

    A vector counts how often each axis word appears in the text.  That is not
    semantics -- it is a deterministic, inspectable stand-in that lets the
    merge, dedup and ranking logic be tested without a 2 GB dependency.  The
    library never constructs one; it is passed in explicitly through the
    ``encoder`` seam.
    """

    AXES = ("dns", "http", "scan", "browsing", "port", "upload")

    def __init__(self, name: str = MODEL_NAME) -> None:
        self._name = name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def dimension(self) -> int:
        return len(self.AXES)

    def encode(self, texts, normalize: bool):
        self.calls += 1
        rows = []
        for text in texts:
            lowered = text.lower()
            row = [1.0 + float(lowered.count(axis)) for axis in self.AXES]
            if normalize:
                length = sum(value * value for value in row) ** 0.5
                row = [value / length for value in row]
            rows.append(row)
        return rows


def document_text(doc_id: str, title: str, signal: str, body: dict[str, str]) -> str:
    sections = "\n".join(
        f"## {name}\n\n{body[name]}\n" for name in (
            "Summary", "What the DPI engine can observe", "Indicators",
            "Benign explanations", "Recommended checks", "References")
    )
    return (
        "---\n"
        f"id: {doc_id}\n"
        f"title: {title}\n"
        "category: protocols\n"
        "version: 1.0\n"
        "updated: 2026-08-27\n"
        "applies_to:\n"
        f"  - {signal}\n"
        "keywords:\n"
        "  - fixture\n"
        "mitre: []\n"
        "severity_hint: info\n"
        "sources:\n"
        "  - Authored for this project.\n"
        "licence: project-authored\n"
        "---\n\n" + sections
    )


def fixture_chunks() -> tuple[KnowledgeChunk, ...]:
    """Three topic documents whose vocabulary maps onto the encoder axes."""
    documents = [
        ("dns-notes", "DNS Notes", "dns_high_volume", "dns"),
        ("http-notes", "HTTP Notes", "plaintext_http", "http"),
        ("scan-notes", "Scan Notes", "scan_port_fanout", "scan port"),
    ]
    chunks: list[KnowledgeChunk] = []
    for doc_id, title, signal, word in documents:
        body = {
            "Summary": f"An overview of {word} behaviour on a network.",
            "What the DPI engine can observe": (
                f"The `protocol`, `dst_port` and `server_name` fields, for {word}."),
            "Indicators": f"Indicators of unusual {word} {word} activity.",
            "Benign explanations": f"Ordinary reasons {word} appears in a capture.",
            "Recommended checks": f"How to check {word} findings before escalating.",
            "References": f"Further reading about {word}.",
        }
        chunks.extend(chunk_document(
            parse_document(document_text(doc_id, title, signal, body),
                           f"protocols/{doc_id}.md")))
    return tuple(chunks)


def build_store(chunks, embedder: EmbeddingModel, name: str = "fixture") -> VectorStore:
    """Index chunks with the supplied embedder -- the step 3 and 4 path, unchanged."""
    store = VectorStore(name)
    embeddings = embedder.embed_chunks(list(chunks))
    store.add_many([VectorRecord(chunk=chunk, embedding=embedding)
                    for chunk, embedding in zip(chunks, embeddings)])
    return store


def keyword_embedder(name: str = MODEL_NAME, encoder: KeywordEncoder | None = None):
    return EmbeddingModel(EmbeddingConfig(model_name=name),
                          encoder=encoder or KeywordEncoder(name))


def flow(flow_id: int, **overrides) -> FlowRecord:
    defaults = dict(
        flow_id=flow_id, protocol=TransportProtocol.TCP, dst_port=443, src_port=50000 + flow_id,
        server_name="www.example.com", application="HTTPS", state="CLASSIFIED",
        verdict="FORWARD", packets_out=8, packets_in=12, bytes_out=900, bytes_in=6400,
        syn_seen=True, syn_ack_seen=True, fin_seen=True, src_ip="host-1", dst_ip="net-1",
    )
    defaults.update(overrides)
    return FlowRecord(**defaults)


def dns_flow(flow_id: int, name: str) -> FlowRecord:
    return flow(flow_id, protocol=TransportProtocol.UDP, dst_port=53, application="DNS",
                server_name=name, syn_seen=False, syn_ack_seen=False, fin_seen=False,
                packets_out=1, packets_in=1, bytes_out=80, bytes_in=180)


def capture(flows: list[FlowRecord], *, dropped_packets: int = 0,
            capture_name: str = "synthetic.pcap") -> CaptureReport:
    applications: dict[str, int] = {}
    for item in flows:
        applications[item.application] = applications.get(item.application, 0) + 1
    return CaptureReport(
        capture_name=capture_name,
        totals=CaptureTotals(
            total_packets=sum(f.packets_out + f.packets_in for f in flows),
            total_bytes=sum(f.bytes_out + f.bytes_in for f in flows),
            tcp_packets=sum(f.packets_out + f.packets_in for f in flows
                            if f.protocol is TransportProtocol.TCP),
            udp_packets=sum(f.packets_out + f.packets_in for f in flows
                            if f.protocol is TransportProtocol.UDP),
            forwarded_packets=sum(f.packets_out + f.packets_in for f in flows),
            dropped_packets=dropped_packets,
            total_flows=len(flows), flows_included=len(flows),
        ),
        application_distribution=applications,
        top_server_names=sorted({f.server_name for f in flows if f.server_name})[:5],
        blocking_rules_active={},
        flows=flows,
        redaction_mode="redact_private",
        notes=[],
    )


def mixed_signals() -> SignalReport:
    """A capture that fires DNS, HTTP and fan-out signals at once."""
    flows = (
        [dns_flow(i, f"unique{i}.example.com") for i in range(8)]
        + [flow(8 + i, dst_port=80, application="HTTP") for i in range(3)]
        + [flow(11 + i, dst_port=9000 + i, application="Unknown", server_name=None,
                syn_ack_seen=False, fin_seen=False, bytes_out=800, bytes_in=0)
           for i in range(10)]
    )
    return extract_signals(capture(flows))


# ===========================================================================
# A. Query construction
# ===========================================================================
def test_query_construction() -> None:
    print("\nA. Query construction")

    report = mixed_signals()
    queries = build_queries(report)

    check("one query per signal, plus the capture query",
          len(queries) == report.signal_count + 1,
          f"{len(queries)} queries for {report.signal_count} signals")
    check("query order follows the signal report order",
          [q.signal_type for q in queries[:-1]]
          == [s.signal_type for s in report.signals])
    check("the capture query comes last", queries[-1].label == CAPTURE_QUERY_LABEL)
    check("every signal query names its signal",
          all(q.signal_id for q in queries[:-1]))
    check("signal ids on queries match the report",
          [q.signal_id for q in queries[:-1]] == [s.signal_id for s in report.signals])

    check("query construction is deterministic",
          [q.text for q in build_queries(report)] == [q.text for q in queries])
    rebuilt = extract_signals(mixed_capture())
    check("query construction is byte-identical for a rebuilt report",
          [q.text for q in build_queries(rebuilt)] == [q.text for q in queries])
    check("embedding text is byte-identical for a rebuilt report",
          [q.embedding_text for q in build_queries(rebuilt)]
          == [q.embedding_text for q in queries])

    # -- content -----------------------------------------------------------
    dns = next(q for q in queries if q.signal_type is SignalType.DNS_HIGH_VOLUME)
    signal = report.by_type(SignalType.DNS_HIGH_VOLUME)
    check("the query carries its signal's topic template",
          QUERY_TEMPLATES[SignalType.DNS_HIGH_VOLUME] in dns.text)
    check("the query carries the signal summary", signal.summary in dns.text)
    check("the query carries numeric evidence", "dns_flow_count=" in dns.text, dns.text)
    check("numeric evidence is rendered in sorted key order",
          dns.text.index("dns_flow_count=") < dns.text.index("total_flow_count="))
    check("every signal type has a query template",
          set(QUERY_TEMPLATES) == set(SignalType))
    check("every query is multi-line and non-trivial",
          all(len(q.text) > 60 for q in queries))

    # String evidence must never be rendered -- this is the injection boundary.
    cardinality = report.by_type(SignalType.DNS_HIGH_CARDINALITY)
    check("the fixture really does carry a hostname in its evidence",
          isinstance(cardinality.evidence.get("top_parent_domain"), str),
          str(cardinality.evidence))
    card_query = next(q for q in queries
                      if q.signal_type is SignalType.DNS_HIGH_CARDINALITY)
    check("string evidence is not rendered into the query",
          cardinality.evidence["top_parent_domain"] not in card_query.text,
          card_query.text)
    check("list evidence is not rendered into the query",
          "distinct_ports" not in "\n".join(q.text for q in queries))

    joined = "\n".join(q.embedding_text for q in queries)
    check("no query contains an IP address", not IPV4.search(joined))
    check("no query contains the capture file name", "synthetic.pcap" not in joined)
    check("no query contains a server name",
          not any(f"unique{i}.example.com" in joined for i in range(8)))

    # -- bge prefix --------------------------------------------------------
    check("every embedding text carries the bge prefix",
          all(q.embedding_text.startswith(BGE_QUERY_PREFIX) for q in queries))
    check("the prefix appears exactly once per query",
          all(q.embedding_text.count(BGE_QUERY_PREFIX) == 1 for q in queries))
    check("the raw query text does not carry the prefix",
          all(not q.text.startswith(BGE_QUERY_PREFIX) for q in queries))
    check("embedding text is exactly prefix + text",
          all(q.embedding_text == BGE_QUERY_PREFIX + q.text for q in queries))
    check("apply_query_prefix adds the prefix once",
          apply_query_prefix("hello") == BGE_QUERY_PREFIX + "hello")
    raises("applying the prefix twice is rejected", RetrievalError,
           lambda: apply_query_prefix(BGE_QUERY_PREFIX + "hello"))

    # -- capture query -----------------------------------------------------
    capture_query = queries[-1]
    check("the capture query reports the flow count",
          f"Flows: {report.flow_count}." in capture_query.text)
    check("the capture query reports protocols", "Protocols by flow count:" in capture_query.text)
    check("the capture query reports ports", "destination ports:" in capture_query.text)
    check("the capture query lists the signals that fired",
          "dns high volume" in capture_query.text, capture_query.text)
    check("the capture query has no signal type", capture_query.signal_type is None)
    check("the capture query has no signal id", capture_query.signal_id is None)
    check("the capture query can be switched off",
          all(q.label != CAPTURE_QUERY_LABEL for q in
              build_queries(report, RetrievalConfig(include_capture_query=False))))

    # -- no secrets --------------------------------------------------------
    os.environ["DPI_RETRIEVAL_TEST_SECRET"] = "sk-must-never-appear"
    try:
        text = "\n".join(q.embedding_text for q in build_queries(report))
        check("no environment value leaks into a query",
              "sk-must-never-appear" not in text)
        for marker in ("sk-", "api_key", "API_KEY", "Bearer ", "password", "payload"):
            check(f"no {marker!r} appears in any query", marker not in text)
        for marker in ("timestamp", "generated_at", "2026-"):
            check(f"no {marker!r} appears in any query", marker not in text)
    finally:
        os.environ.pop("DPI_RETRIEVAL_TEST_SECRET", None)

    # -- the guard actually guards ----------------------------------------
    tampered = Signal(
        signal_id="dns_high_volume#0123456789ab",
        signal_type=SignalType.DNS_HIGH_VOLUME,
        severity=Severity.LOW,
        confidence=0.5,
        summary="Queries were sent to evil.example.com repeatedly during the capture.",
        does_not_prove="Nothing at all is proven by this fixture.",
        evidence={"dns_flow_count": 5},
        flow_ids=(),
        source_fields=(SourceField.DST_PORT,),
    )
    tampered_report = SignalReport(
        generated_from="test", capture_name="c", redaction_mode="none",
        flow_count=5, total_flow_count=5, signal_count=1, signals=(tampered,))
    raises("a hostname reaching a query is refused, not embedded", RetrievalError,
           lambda: build_queries(tampered_report))

    # -- model validation --------------------------------------------------
    valid = dict(label="capture", signal_type=None, signal_id=None,
                 text="a query about network security knowledge",
                 embedding_text=BGE_QUERY_PREFIX + "a query about network security knowledge")

    def variant(**changes):
        data = dict(valid)
        data.update(changes)
        return lambda: SignalQuery(**data)

    raises("a malformed query label is rejected", ValueError, variant(label="whatever"))
    raises("a mismatched embedding text is rejected", ValueError,
           variant(embedding_text="not the prefix plus text"))
    raises("a missing prefix is rejected", ValueError, variant(embedding_text=valid["text"]))
    raises("a capture query with a signal id is rejected", ValueError,
           variant(signal_id="dns_high_volume#0123456789ab"))
    raises("a signal query with no signal id is rejected", ValueError,
           variant(label="signal:dns_high_volume", signal_type=SignalType.DNS_HIGH_VOLUME))
    raises("a label that disagrees with the signal type is rejected", ValueError,
           variant(label="signal:plaintext_http", signal_type=SignalType.DNS_HIGH_VOLUME,
                   signal_id="dns_high_volume#0123456789ab"))
    raises("an unexpected field on a query is rejected", ValueError, variant(extra="x"))


def mixed_capture() -> CaptureReport:
    flows = (
        [dns_flow(i, f"unique{i}.example.com") for i in range(8)]
        + [flow(8 + i, dst_port=80, application="HTTP") for i in range(3)]
        + [flow(11 + i, dst_port=9000 + i, application="Unknown", server_name=None,
                syn_ack_seen=False, fin_seen=False, bytes_out=800, bytes_in=0)
           for i in range(10)]
    )
    return capture(flows)


# ===========================================================================
# B. Empty and edge cases
# ===========================================================================
def test_empty_cases() -> None:
    print("\nB. Empty and edge cases")

    empty_report = extract_signals(capture([]))
    check("an empty capture fires no signals", empty_report.signal_count == 0)
    check("an empty capture produces no queries", build_queries(empty_report) == ())

    chunks = fixture_chunks()
    embedder = keyword_embedder()
    store = build_store(chunks, embedder)

    result = retrieve_for_signals(empty_report, store, embedder)
    check("an empty capture still produces a valid report",
          isinstance(result, RetrievalReport))
    check("it retrieves nothing", result.chunk_count == 0 and result.chunks == ())
    check("it ran no queries", result.query_count == 0)
    check("it says why", any("No signals" in note for note in result.notes),
          str(result.notes))
    check("it still records the schema version",
          result.schema_version == RETRIEVAL_SCHEMA_VERSION == "1.0")
    check("it still records the model", result.model_name == MODEL_NAME)

    # Only the baseline signal.
    baseline_report = extract_signals(capture([flow(0)]))
    check("a minimal capture fires only the baseline signal",
          baseline_report.types() == ("baseline_web_browsing",),
          str(baseline_report.types()))
    baseline_queries = build_queries(baseline_report)
    check("the baseline capture still builds two queries", len(baseline_queries) == 2)
    baseline_result = retrieve_for_signals(baseline_report, store, embedder)
    check("the baseline capture still retrieves something",
          baseline_result.chunk_count > 0)

    # top_k and threshold edges.
    report = mixed_signals()
    check("per_query_top_k=0 retrieves nothing",
          retrieve_for_signals(report, store, embedder,
                               RetrievalConfig(per_query_top_k=0)).chunk_count == 0)
    check("final_top_k=0 retrieves nothing",
          retrieve_for_signals(report, store, embedder,
                               RetrievalConfig(final_top_k=0)).chunk_count == 0)
    check("a threshold above every score retrieves nothing",
          retrieve_for_signals(report, store, embedder,
                               RetrievalConfig(min_similarity=1.0)).chunk_count == 0)

    raises("a negative per_query_top_k is rejected", ValueError,
           lambda: RetrievalConfig(per_query_top_k=-1))
    raises("a negative final_top_k is rejected", ValueError,
           lambda: RetrievalConfig(final_top_k=-1))
    raises("a zero max_per_document is rejected", ValueError,
           lambda: RetrievalConfig(max_per_document=0))
    raises("a negative max_per_document is rejected", ValueError,
           lambda: RetrievalConfig(max_per_document=-1))
    check("max_per_document=None disables the cap",
          RetrievalConfig(max_per_document=None).max_per_document is None)
    raises("an out-of-range min_similarity is rejected", ValueError,
           lambda: RetrievalConfig(min_similarity=1.5))
    raises("searching an empty store is refused", RetrievalError,
           lambda: retrieve_for_signals(report, VectorStore("empty"), embedder))


# ===========================================================================
# C. Retrieval mechanics
# ===========================================================================
def test_retrieval() -> None:
    print("\nC. Retrieval mechanics")

    chunks = fixture_chunks()
    embedder = keyword_embedder()
    store = build_store(chunks, embedder)
    report = mixed_signals()

    result = retrieve_for_signals(report, store, embedder)
    check("retrieval returns chunks", result.chunk_count > 0)
    check("it never exceeds final_top_k", result.chunk_count <= 8)
    check("every similarity is within [-1, 1]",
          all(-1.0 <= c.similarity <= 1.0 for c in result.chunks))
    check("results are ordered by similarity descending",
          all(a.similarity >= b.similarity
              for a, b in zip(result.chunks, result.chunks[1:])),
          str([round(c.similarity, 4) for c in result.chunks]))
    check("ranks are contiguous from zero",
          [c.rank for c in result.chunks] == list(range(result.chunk_count)))
    check("no chunk appears twice",
          len(set(result.chunk_ids())) == result.chunk_count)

    # An exact match scores 1.0: embed a chunk's own indexed text as a query.
    from ai.rag.embeddings import build_embedding_input

    target = chunks[0]
    exact = embedder.embed_texts([build_embedding_input(target)])[0]
    hits = store.search(exact, top_k=6)
    check("an exact vector scores 1.0", abs(hits[0].similarity - 1.0) < 1e-9,
          str(hits[0].similarity))
    # The keyword encoder deliberately collides: several fixture chunks share a
    # vector, so the chunk itself is among the perfect matches rather than
    # necessarily first, and the store's chunk_id tie-break decides the order.
    perfect = [h for h in hits if abs(h.similarity - 1.0) < 1e-9]
    check("a chunk's own vector retrieves it among the perfect matches",
          target.chunk_id in {h.chunk_id for h in perfect},
          str([h.chunk_id for h in perfect]))
    check("perfect matches are ordered by chunk_id",
          [h.chunk_id for h in perfect] == sorted(h.chunk_id for h in perfect))

    # Threshold and top_k.
    tight = retrieve_for_signals(report, store, embedder,
                                 RetrievalConfig(final_top_k=3))
    check("final_top_k limits the result", tight.chunk_count == 3)
    check("the smaller result is a prefix of the larger",
          tight.chunk_ids() == result.chunk_ids()[:3],
          f"{tight.chunk_ids()} vs {result.chunk_ids()[:3]}")

    floor = min(c.similarity for c in result.chunks)
    filtered = retrieve_for_signals(report, store, embedder,
                                    RetrievalConfig(min_similarity=floor + 1e-6))
    check("a similarity floor removes the weakest results",
          filtered.chunk_count < result.chunk_count,
          f"{filtered.chunk_count} vs {result.chunk_count}")
    check("everything kept is above the floor",
          all(c.similarity > floor for c in filtered.chunks))

    # Repeatability.
    again = retrieve_for_signals(report, store, embedder)
    check("repeated retrieval returns identical chunk ids",
          again.chunk_ids() == result.chunk_ids())
    check("repeated retrieval returns identical scores",
          [c.similarity for c in again.chunks] == [c.similarity for c in result.chunks])
    check("repeated retrieval serialises identically", again.to_json() == result.to_json())
    check("a rebuilt store gives the same result",
          retrieve_for_signals(report, build_store(chunks, keyword_embedder()),
                               keyword_embedder()).chunk_ids() == result.chunk_ids())

    # Ties: two documents with identical text score identically.
    twin_a = chunk_document(parse_document(
        document_text("aaa-twin", "AAA Twin", "dns_high_volume",
                      {name: "Identical body text about dns behaviour." for name in (
                          "Summary", "What the DPI engine can observe", "Indicators",
                          "Benign explanations", "Recommended checks", "References")}),
        "protocols/aaa-twin.md"))
    twin_b = chunk_document(parse_document(
        document_text("zzz-twin", "AAA Twin", "dns_high_volume",
                      {name: "Identical body text about dns behaviour." for name in (
                          "Summary", "What the DPI engine can observe", "Indicators",
                          "Benign explanations", "Recommended checks", "References")}),
        "protocols/zzz-twin.md"))
    twin_embedder = keyword_embedder()
    twin_store = build_store(list(twin_a) + list(twin_b), twin_embedder, "twins")
    twin_result = retrieve_for_signals(report, twin_store, twin_embedder,
                                       RetrievalConfig(max_per_document=None, final_top_k=6))
    tied = [c for c in twin_result.chunks
            if abs(c.similarity - twin_result.chunks[0].similarity) < 1e-12]
    check("identical documents produce tied scores", len(tied) >= 2, str(len(tied)))
    check("ties break on chunk_id ascending",
          [c.chunk_id for c in tied] == sorted(c.chunk_id for c in tied),
          str([c.chunk_id for c in tied]))
    check("tie ordering is repeatable",
          retrieve_for_signals(report, twin_store, twin_embedder,
                               RetrievalConfig(max_per_document=None,
                                               final_top_k=6)).chunk_ids()
          == twin_result.chunk_ids())


# ===========================================================================
# D. Signal-aware retrieval, dedup and diversity
# ===========================================================================
def test_signal_aware() -> None:
    print("\nD. Signal-aware retrieval")

    chunks = fixture_chunks()
    embedder = keyword_embedder()
    store = build_store(chunks, embedder)
    report = mixed_signals()

    result = retrieve_for_signals(report, store, embedder,
                                  RetrievalConfig(per_query_top_k=4, final_top_k=12,
                                                  max_per_document=None))

    check("more than one query contributed",
          len({label for c in result.chunks for label in c.matched_query_labels}) > 1)
    check("more than one document was retrieved",
          len(set(result.document_ids())) > 1, str(result.document_ids()))
    check("every chunk records at least one matching query",
          all(c.matched_query_labels for c in result.chunks))
    check("every matched label was actually run",
          {label for c in result.chunks for label in c.matched_query_labels}
          <= {q.label for q in result.queries})

    # Deduplication.
    shared = [c for c in result.chunks if len(c.matched_query_labels) > 1]
    check("some chunk was retrieved by more than one query", shared != [],
          "the fixture should produce overlap")
    check("a chunk retrieved twice appears once",
          len(set(result.chunk_ids())) == result.chunk_count)
    for chunk in shared:
        check(f"{chunk.chunk_id[:38]}: score is the best of its queries",
              abs(chunk.similarity - max(chunk.per_query_similarity.values())) < 1e-12,
              f"{chunk.similarity} vs {chunk.per_query_similarity}")
    check("per-query scores are recorded for every match",
          all(set(c.per_query_similarity) == set(c.matched_query_labels)
              for c in result.chunks))
    check("matched signal types mirror the matched labels",
          all(tuple(t.value for t in c.matched_signal_types)
              == tuple(sorted(label.split(":", 1)[1] for label in c.matched_query_labels
                              if label != CAPTURE_QUERY_LABEL))
              for c in result.chunks))
    check("matched labels are sorted",
          all(list(c.matched_query_labels) == sorted(c.matched_query_labels)
              for c in result.chunks))

    # for_signal() finds what a given signal pulled in.
    dns_hits = result.for_signal(SignalType.DNS_HIGH_VOLUME)
    check("the DNS signal retrieved something", dns_hits != ())
    check("its hits all record that signal",
          all(SignalType.DNS_HIGH_VOLUME in c.matched_signal_types for c in dns_hits))
    check("the DNS signal favours the DNS document",
          dns_hits[0].document_id == "dns-notes", dns_hits[0].document_id)

    http_hits = result.for_signal(SignalType.PLAINTEXT_HTTP)
    check("the HTTP signal retrieved something", http_hits != ())
    check("the HTTP signal favours the HTTP document",
          http_hits[0].document_id == "http-notes", http_hits[0].document_id)

    # Adding an unrelated signal must not reorder unrelated results.
    with_extra = extract_signals(capture(
        list(mixed_capture().flows)
        + [flow(100 + i, protocol=TransportProtocol.UDP, dst_port=443,
                application="QUIC", syn_seen=False, syn_ack_seen=False, fin_seen=False)
           for i in range(2)]))
    check("the extra capture fires an additional signal",
          "quic_present" in with_extra.types() and
          "quic_present" not in report.types())
    extra_result = retrieve_for_signals(with_extra, store, embedder,
                                        RetrievalConfig(per_query_top_k=4, final_top_k=12,
                                                        max_per_document=None))
    common = [c for c in extra_result.chunks if c.chunk_id in set(result.chunk_ids())]
    original_order = [c.chunk_id for c in result.chunks if c.chunk_id in
                      {c.chunk_id for c in common}]
    check("adding a signal does not reshuffle the chunks it does not touch",
          [c.chunk_id for c in common
           if SignalType.QUIC_PRESENT not in c.matched_signal_types]
          == [cid for cid in original_order
              if SignalType.QUIC_PRESENT not in
              next(c for c in extra_result.chunks if c.chunk_id == cid
                   ).matched_signal_types],
          "unrelated ordering changed")

    # Diversity cap.
    capped = retrieve_for_signals(report, store, embedder,
                                  RetrievalConfig(per_query_top_k=6, final_top_k=12,
                                                  max_per_document=2))
    counts: dict[str, int] = {}
    for chunk in capped.chunks:
        counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1
    check("the diversity cap is honoured", all(count <= 2 for count in counts.values()),
          str(counts))
    check("the cap is reported in the notes",
          any("diversity cap" in note for note in capped.notes), str(capped.notes))
    check("disabling the cap allows more from one document",
          max(_document_counts(retrieve_for_signals(
              report, store, embedder,
              RetrievalConfig(per_query_top_k=6, final_top_k=12,
                              max_per_document=None))).values()) > 2)
    check("the capped result is still deterministic",
          retrieve_for_signals(report, store, embedder,
                               RetrievalConfig(per_query_top_k=6, final_top_k=12,
                                               max_per_document=2)).chunk_ids()
          == capped.chunk_ids())


def _document_counts(result: RetrievalReport) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in result.chunks:
        counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1
    return counts


# ===========================================================================
# E. Provenance and the result models
# ===========================================================================
def test_provenance() -> None:
    print("\nE. Provenance")

    chunks = fixture_chunks()
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    embedder = keyword_embedder()
    store = build_store(chunks, embedder)
    result = retrieve_for_signals(mixed_signals(), store, embedder)

    for retrieved in result.chunks:
        source = by_id[retrieved.chunk_id]
        ok = (
            retrieved.document_id == source.document_id
            and retrieved.title == source.title
            and retrieved.category is source.category
            and retrieved.section == source.section
            and retrieved.text == source.text
            and retrieved.heading_path == source.heading_path
            and retrieved.licence == source.licence
            and retrieved.sources == list(source.sources)
        )
        check(f"{retrieved.citation()}: provenance is preserved", ok)

    top = result.chunks[0]
    check("a retrieved chunk can cite its document and section",
          top.document_id in top.citation() and top.section in top.citation())
    check("the chunk itself is carried, not copied field by field",
          isinstance(top.chunk, KnowledgeChunk))
    check("the vector is not carried into the retrieval result",
          "vector" not in RetrievedChunk.model_fields)
    check("metadata() omits the chunk text", "text" not in top.metadata())
    check("metadata() records the rank and score",
          top.metadata()["rank"] == 0 and top.metadata()["similarity"] == top.similarity)
    check("declares_matched_signal reports the corpus link",
          isinstance(top.declares_matched_signal(), bool))
    check("a DNS hit on the DNS document declares its signal",
          result.for_signal(SignalType.DNS_HIGH_VOLUME)[0].declares_matched_signal() is True)

    check("the report names what it came from",
          "SignalReport" in result.generated_from)
    check("the report names the capture", result.capture_name == "synthetic.pcap")
    check("the report records the model and dimension",
          result.model_name == MODEL_NAME and result.dimension == 6)
    check("the report records the parameters used",
          set(result.parameters) == {"per_query_top_k", "final_top_k", "min_similarity",
                                     "max_per_document", "include_capture_query",
                                     "affinity", "query_style"},
          str(sorted(result.parameters)))
    check("the report exposes distinct document ids",
          set(result.document_ids()) <= {c.document_id for c in chunks})
    check("to_json omits chunk text by default",
          "text" not in json.loads(result.to_json())["chunks"][0])
    check("to_json can include chunk text on request",
          "text" in json.loads(result.to_json(include_text=True))["chunks"][0])

    # -- model rejection cases --------------------------------------------
    good = result.chunks[0]

    def variant(**changes):
        data = dict(chunk=good.chunk, similarity=good.similarity, rank=good.rank,
                    matched_signal_types=good.matched_signal_types,
                    matched_query_labels=good.matched_query_labels,
                    per_query_similarity=dict(good.per_query_similarity))
        data.update(changes)
        return lambda: RetrievedChunk(**data)

    raises("a NaN similarity is rejected", ValueError, variant(similarity=float("nan")))
    raises("an infinite similarity is rejected", ValueError,
           variant(similarity=float("inf")))
    raises("a similarity above 1 is rejected", ValueError, variant(similarity=1.5))
    raises("a similarity below -1 is rejected", ValueError, variant(similarity=-1.5))
    raises("a negative rank is rejected", ValueError, variant(rank=-1))
    raises("no matched query label is rejected", ValueError,
           variant(matched_query_labels=(), per_query_similarity={}))
    raises("unsorted matched labels are rejected", ValueError,
           variant(matched_query_labels=("signal:plaintext_http", "capture"),
                   per_query_similarity={"capture": 0.5, "signal:plaintext_http": 0.4},
                   matched_signal_types=(SignalType.PLAINTEXT_HTTP,)))
    raises("an invalid matched label is rejected", ValueError,
           variant(matched_query_labels=("not a label",),
                   per_query_similarity={"not a label": 0.5},
                   matched_signal_types=()))
    raises("per-query keys that disagree with the labels are rejected", ValueError,
           variant(per_query_similarity={"capture": 0.5}))
    raises("a similarity that is not the best per-query score is rejected", ValueError,
           variant(similarity=0.1))
    raises("signal types that disagree with the labels are rejected", ValueError,
           variant(matched_signal_types=(SignalType.QUIC_PRESENT,)))
    raises("an unexpected field on a retrieved chunk is rejected", ValueError,
           variant(extra="value"))
    check("retrieved chunks are immutable", RetrievedChunk.model_config["frozen"] is True)

    def mutate() -> None:
        result.chunks[0].rank = 5  # type: ignore[misc]

    raises("a retrieved chunk cannot be mutated", ValueError, mutate)

    # -- report-level rules ------------------------------------------------
    base = dict(generated_from="x", capture_name="c", model_name=MODEL_NAME,
                dimension=6, query_count=len(result.queries), queries=result.queries)
    raises("a duplicated chunk in the report is rejected", ValueError,
           lambda: RetrievalReport(**base, chunk_count=2,
                                   chunks=(result.chunks[0], result.chunks[0])))
    raises("non-contiguous ranks are rejected", ValueError,
           lambda: RetrievalReport(**base, chunk_count=1,
                                   chunks=(result.chunks[1],)))
    raises("a chunk_count that disagrees is rejected", ValueError,
           lambda: RetrievalReport(**base, chunk_count=9, chunks=result.chunks))
    raises("a chunk citing an unknown query is rejected", ValueError,
           lambda: RetrievalReport(generated_from="x", capture_name="c",
                                   model_name=MODEL_NAME, dimension=6,
                                   query_count=0, queries=(),
                                   chunk_count=1, chunks=(result.chunks[0],)))


# ===========================================================================
# F. Privacy
# ===========================================================================
def test_privacy() -> None:
    print("\nF. Privacy")

    chunks = fixture_chunks()
    embedder = keyword_embedder()
    store = build_store(chunks, embedder)

    os.environ["DPI_RETRIEVAL_TEST_SECRET"] = "sk-must-never-appear"
    try:
        result = retrieve_for_signals(mixed_signals(), store, embedder)
        serialized = result.to_json(include_text=True)
        check("no environment value leaks into a retrieval report",
              "sk-must-never-appear" not in serialized)
        for marker in ("sk-", "api_key", "API_KEY", "Bearer ", "password"):
            check(f"no {marker!r} appears in a retrieval report", marker not in serialized)
    finally:
        os.environ.pop("DPI_RETRIEVAL_TEST_SECRET", None)

    check("no IP address appears in a retrieval report", not IPV4.search(serialized),
          str(IPV4.findall(serialized)[:3]))
    check("no capture hostname appears in a retrieval report",
          not any(f"unique{i}.example.com" in serialized for i in range(8)))
    check("no timestamp key appears in a retrieval report",
          not any(key in serialized.lower()
                  for key in ("timestamp", "generated_at", "retrieved_at", "created_at")))
    check("the report model has no time field",
          not any("_at" in name for name in RetrievalReport.model_fields))

    source_code = Path("ai/rag/retrieval.py").read_text(encoding="utf-8")
    for banned in ("import time", "import random", "import uuid", "datetime",
                   "monotonic", "uuid4"):
        check(f"retrieval never uses {banned!r}", banned not in source_code)
    for banned in ("requests", "urllib", "httpx", "http.client", "socket",
                   "openai", "groq", "langchain", "faiss", "chromadb"):
        check(f"retrieval never references {banned!r}", banned not in source_code)
    check("retrieval does not import numpy",
          "numpy" not in source_code and "import numpy" not in source_code)
    check("retrieval defines no similarity function of its own",
          "def cosine" not in source_code and "def _cosine" not in source_code)
    check("retrieval delegates search to the vector store",
          "store.search(" in source_code)
    check("the openai SDK is not imported", "openai" not in sys.modules)
    check("no vector database library is imported",
          not any(name in sys.modules for name in ("faiss", "chromadb", "qdrant_client")))
    check("no LangChain library is imported",
          not any(name.startswith("langchain") for name in sys.modules))

    # Only chunk text from the reviewed corpus is carried out.
    corpus_text = {chunk.text for chunk in chunks}
    check("retrieved text comes only from the indexed corpus",
          all(c.text in corpus_text for c in result.chunks))


# ===========================================================================
# G. Model compatibility
# ===========================================================================
def test_model_compatibility() -> None:
    print("\nG. Model compatibility")

    chunks = fixture_chunks()
    embedder = keyword_embedder()
    store = build_store(chunks, embedder)
    report = mixed_signals()

    check("a matching model retrieves normally",
          retrieve_for_signals(report, store, embedder).chunk_count > 0)

    other = keyword_embedder("other/model-6d", KeywordEncoder("other/model-6d"))
    raises("a different embedding model is refused", ModelMismatchError,
           lambda: retrieve_for_signals(report, store, other))

    class NarrowEncoder(KeywordEncoder):
        AXES = ("dns", "http", "scan")

    narrow = EmbeddingModel(EmbeddingConfig(model_name=MODEL_NAME),
                            encoder=NarrowEncoder(MODEL_NAME))
    raises("a query of the wrong width is refused", ModelMismatchError,
           lambda: retrieve_for_signals(report, store, narrow))

    check("the store's own model guard still applies",
          store.model_name == MODEL_NAME and store.dimension == 6)
    raises("an empty store is refused before anything is embedded", RetrievalError,
           lambda: retrieve_for_signals(report, VectorStore("empty"), embedder))


# ===========================================================================
# H. No network
# ===========================================================================
def test_no_network() -> None:
    print("\nH. No network")

    chunks = fixture_chunks()
    embedder = keyword_embedder()
    store = build_store(chunks, embedder)
    report = mixed_signals()

    real_socket, real_connect = socket.socket, socket.create_connection

    def refuse(*args, **kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("retrieval attempted a network connection")

    socket.socket, socket.create_connection = refuse, refuse  # type: ignore[assignment]
    try:
        queries = build_queries(report)
        check("query construction makes no network call", len(queries) > 0)
        offline = retrieve_for_signals(report, store, embedder)
        check("retrieval against a loaded model makes no network call",
              offline.chunk_count > 0)
    finally:
        socket.socket, socket.create_connection = real_socket, real_connect

    check("retrieval works entirely with synthetic embeddings",
          retrieve_for_signals(report, store, embedder).chunk_count > 0)


# ===========================================================================
# I. Integration -- the real corpus and the real model
# ===========================================================================
def test_integration() -> None:
    print(f"\nI. Integration -- real corpus with {DEFAULT_MODEL} (optional)")

    labels = (
        "the real corpus indexes for retrieval",
        "a DNS-heavy capture retrieves chunks",
        "every retrieved chunk id is real",
        "every retrieved chunk keeps its provenance",
        "every similarity is within [-1, 1]",
        "retrieval on the real index is deterministic",
        "the DNS signal retrieves DNS knowledge",
        "results respect the diversity cap",
        "no address or secret appears in the real report",
    )

    if not sentence_transformers_available():
        for label in labels:
            skip(label, "sentence-transformers is not installed "
                        "(pip install -r requirements-rag.txt)")
        return

    embedder = EmbeddingModel(EmbeddingConfig())
    try:
        embedder.load()
    except ModelUnavailableError as exc:
        for label in labels:
            skip(label, f"{DEFAULT_MODEL} could not be loaded: {str(exc)[:100]}")
        return

    corpus = load_corpus()
    chunks = chunk_corpus(corpus)
    store = build_store(chunks, embedder, "knowledge")
    check(labels[0], store.count() == len(chunks) == 37, str(store.count()))

    report = extract_signals(capture(
        [dns_flow(i, f"k7f2q9x4m1z8b3v6n5c{i}.tunnel.example") for i in range(8)]
        + [flow(8 + i) for i in range(4)]))
    result = retrieve_for_signals(report, store, embedder)

    known = {chunk.chunk_id for chunk in chunks}
    check(labels[1], result.chunk_count > 0, str(result.chunk_count))
    check(labels[2], all(c.chunk_id in known for c in result.chunks))
    check(labels[3], all(c.document_id and c.section and c.title and c.licence
                         for c in result.chunks))
    check(labels[4], all(-1.0 <= c.similarity <= 1.0 for c in result.chunks))
    check(labels[5], retrieve_for_signals(report, store, embedder).to_json()
          == result.to_json())

    dns_hits = result.for_signal(SignalType.DNS_HIGH_VOLUME)
    check(labels[6], any("dns" in c.document_id for c in dns_hits),
          str([c.document_id for c in dns_hits]))
    check(labels[7], all(count <= 2 for count in _document_counts(result).values()),
          str(_document_counts(result)))
    serialized = result.to_json(include_text=True)
    check(labels[8], not IPV4.search(serialized) and "sk-" not in serialized)

    print(f"        {result.query_count} queries -> {result.chunk_count} chunks "
          f"from {len(set(result.document_ids()))} documents")
    for hit in result.chunks[:4]:
        print(f"        #{hit.rank}  {hit.similarity:.4f}  {hit.citation()}")


# ===========================================================================
def main() -> int:
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print("RAG step 6 -- signal-driven retrieval")

    test_query_construction()
    test_empty_cases()
    test_retrieval()
    test_signal_aware()
    test_provenance()
    test_privacy()
    test_model_compatibility()
    test_no_network()
    test_integration()

    total = _passed + _failed
    suffix = f", {_skipped} skipped" if _skipped else ""
    print(f"\n{_passed}/{total} checks passed{suffix}")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
