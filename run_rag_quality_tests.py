"""Test runner for retrieval *quality* -- RAG step 9.

What separates this suite from the others
-----------------------------------------
Steps 4 and 6 test that retrieval is correct: the arithmetic is cosine, the
ranking is stable, nothing leaks, nothing is invented.  All of that can be true
of a retriever that consistently returns the wrong document.  This suite tests
the part that decides *which* knowledge is appropriate for an observation --
:mod:`ai.rag.affinity`, its use in ranking, the query framing that feeds it, and
the request-size guard that keeps the result sendable.

Three tiers
-----------
**Policy tier** -- pure functions.  Compatibility classification, its notes and
its ordering.  No model, no corpus, no index.

**Corpus tier** -- the real ``knowledge/`` front matter against the real
:mod:`evaluation.cases` labels, with no embedding model.  These are the checks
that matter most: they assert that the compatibility rule agrees with the
hand-written relevance labels on every case, which is what makes the rule a
statement about the corpus rather than a fit to a leaderboard.  If someone adds
a document, or a case, whose declared scope and whose labels disagree, this
suite says so.

**Mechanics tier** -- ranking behaviour through the ``encoder`` seam, with a
keyword encoder defined in this file.  Its vectors are keyword counts, so which
chunk wins is worked out by hand; the library has no fallback encoder and can
never select it.

Nothing here needs ``BAAI/bge-small-en-v1.5``, and nothing here measures
semantic quality -- that is what ``run_rag_evaluation.py`` is for, on a machine
where the model loads.

Run::

    python run_rag_quality_tests.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys

# Import the package so its console-encoding fix is applied before any output.
import dpi  # noqa: F401
from ai.config import AIConfig
from ai.llm_client import FailureReason, estimate_request_tokens
from ai.analyzer import explain_failure
from ai.providers import Provider, get_provider_spec
from ai.rag.affinity import (
    TIER,
    AffinityMode,
    Compatibility,
    assess,
    assess_many,
)
from ai.rag.chunking import chunk_corpus, chunk_document
from ai.rag.documents import load_corpus, parse_document
from ai.rag.embeddings import EmbeddingConfig, EmbeddingModel
from ai.rag.retrieval import (
    DEFAULT_QUERY_STYLE,
    QUERY_LEAD_IN,
    RetrievalConfig,
    build_queries,
    retrieve_for_signals,
)
from ai.rag.signals import extract_signals
from ai.rag.vector_store import VectorRecord, VectorStore
from ai.schemas import CaptureReport, CaptureTotals, FlowRecord, TransportProtocol
from evaluation.cases import CASES

_passed = 0
_failed = 0
_skipped = 0

MODEL_NAME = "test/quality-6d"


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
    except Exception as exc:  # noqa: BLE001
        check(label, False, f"raised {type(exc).__name__} instead: {exc}")
    else:
        check(label, False, "nothing was raised")


# ===========================================================================
# Fixtures
# ===========================================================================
class KeywordEncoder:
    """Keyword-count vectors, so every ranking outcome is predictable by hand.

    Not semantics.  A deterministic, inspectable stand-in that lets the ranking
    policy be tested without a 2 GB dependency, passed in explicitly through
    the ``encoder`` seam.
    """

    AXES = ("dns", "http", "scan", "browsing", "port", "upload")

    def __init__(self, name: str = MODEL_NAME) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def dimension(self) -> int:
        return len(self.AXES)

    def encode(self, texts, normalize: bool):
        rows = []
        for text in texts:
            lowered = text.lower()
            row = [1.0 + float(lowered.count(axis)) for axis in self.AXES]
            if normalize:
                length = sum(value * value for value in row) ** 0.5
                row = [value / length for value in row]
            rows.append(row)
        return rows


def document_text(doc_id: str, title: str, applies_to: list[str], word: str) -> str:
    scope = "\n".join(f"  - {name}" for name in applies_to) or "  []"
    listed = f"applies_to:\n{scope}" if applies_to else "applies_to: []"
    return f"""\
---
id: {doc_id}
title: {title}
category: protocols
version: 1.0
updated: 2026-08-27
{listed}
keywords:
  - {word}
mitre: []
severity_hint: info
sources:
  - Authored for this project.
licence: project-authored
---

## Summary

An overview of {word} behaviour on a network, in general terms.

## What the DPI engine can observe

The `protocol`, `dst_port` and `server_name` fields, for {word} traffic.

## Indicators

Indicators of unusual {word} {word} {word} activity worth a second look.

## Benign explanations

Ordinary reasons {word} appears in a capture of normal traffic.

## Recommended checks

How to check a {word} finding before escalating it to anyone.

## References

Further reading about {word} and related subjects.
"""


def build_store(chunks, embedder: EmbeddingModel, name: str = "quality") -> VectorStore:
    store = VectorStore(name)
    store.add_many([VectorRecord(chunk=chunk, embedding=embedding)
                    for chunk, embedding
                    in zip(chunks, embedder.embed_chunks(list(chunks)))])
    return store


def flow(flow_id: int, **overrides) -> FlowRecord:
    defaults = dict(
        flow_id=flow_id, protocol=TransportProtocol.TCP, dst_port=443,
        src_port=50000 + flow_id, server_name="www.example.com", application="HTTPS",
        state="CLASSIFIED", verdict="FORWARD", packets_out=8, packets_in=12,
        bytes_out=900, bytes_in=6400, syn_seen=True, syn_ack_seen=True,
        fin_seen=True, src_ip="host-1", dst_ip="net-1",
    )
    defaults.update(overrides)
    return FlowRecord(**defaults)


def capture(flows: list[FlowRecord], name: str = "quality.pcap") -> CaptureReport:
    applications: dict[str, int] = {}
    for item in flows:
        applications[item.application] = applications.get(item.application, 0) + 1
    return CaptureReport(
        capture_name=name,
        totals=CaptureTotals(
            total_packets=sum(f.packets_out + f.packets_in for f in flows),
            total_bytes=sum(f.bytes_out + f.bytes_in for f in flows),
            tcp_packets=sum(f.packets_out + f.packets_in for f in flows
                            if f.protocol is TransportProtocol.TCP),
            udp_packets=sum(f.packets_out + f.packets_in for f in flows
                            if f.protocol is TransportProtocol.UDP),
            forwarded_packets=sum(f.packets_out + f.packets_in for f in flows),
            dropped_packets=0,
            total_flows=len(flows), flows_included=len(flows),
        ),
        application_distribution=dict(sorted(applications.items())),
        top_server_names=sorted({f.server_name for f in flows if f.server_name})[:5],
        blocking_rules_active={}, flows=flows,
        redaction_mode="redact_private", notes=[],
    )


# ===========================================================================
# A. Compatibility policy
# ===========================================================================
def test_policy() -> None:
    print("\nA. Compatibility policy")

    declared = assess(["dns_high_volume", "quic_present"], ["quic_present"])
    check("a declared signal makes a note compatible",
          declared.compatibility is Compatibility.DECLARED
          and declared.declared_matches == ("quic_present",),
          str(declared))

    undeclared = assess(["dns_high_volume", "dns_anomalous_label"],
                        ["baseline_web_browsing"])
    check("a scoped note whose signals did not fire is undeclared",
          undeclared.compatibility is Compatibility.UNDECLARED
          and undeclared.declared_matches == (),
          str(undeclared))

    unscoped = assess([], ["baseline_web_browsing"])
    check("a note that declares nothing is unscoped, not demoted",
          unscoped.compatibility is Compatibility.UNSCOPED)

    nothing_fired = assess(["dns_high_volume"], [])
    check("when no signal fired nothing is demoted",
          nothing_fired.compatibility is Compatibility.UNSCOPED,
          str(nothing_fired))

    check("the tiers order compatible before unscoped before undeclared",
          TIER[Compatibility.DECLARED] < TIER[Compatibility.UNSCOPED]
          < TIER[Compatibility.UNDECLARED])

    check("every verdict carries a reason a person can read",
          all(len(v.note) >= 20 and len(v.note) <= 300
              for v in (declared, undeclared, unscoped, nothing_fired)),
          str([len(v.note) for v in (declared, undeclared, unscoped, nothing_fired)]))

    check("the reason names the signals it is talking about",
          "quic_present" in declared.note
          and "dns_high_volume" in undeclared.note)

    check("only a tiered verdict reports itself as adjusted",
          declared.adjusted and undeclared.adjusted and not unscoped.adjusted)

    a = assess(["b", "a", "c"], ["c", "a"])
    b = assess(["c", "b", "a"], ["a", "c"])
    check("classification does not depend on input order",
          a == b and a.declared_matches == ("a", "c"), str((a, b)))

    many = assess(["s1", "s2", "s3", "s4", "s5"], ["z"])
    check("a long declared scope is abbreviated, not dumped",
          "and 2 more" in many.note, many.note)

    mapping = assess_many({"x": ["dns_high_volume"], "y": []}, ["dns_high_volume"])
    check("assess_many classifies each id independently",
          mapping["x"].compatibility is Compatibility.DECLARED
          and mapping["y"].compatibility is Compatibility.UNSCOPED)

    check("an empty signal name is ignored rather than matched",
          assess(["", "dns_high_volume"], ["", ""]).compatibility
          is Compatibility.UNSCOPED)


# ===========================================================================
# B. The rule against the real corpus and the real labels
# ===========================================================================
def test_corpus_agreement() -> None:
    print("\nB. Compatibility versus the hand-written relevance labels")
    print("   (the real corpus front matter; no embedding model involved)")

    corpus = load_corpus()
    scope = {document.metadata.id: list(document.metadata.applies_to)
             for document in corpus}
    check("the corpus loaded", len(scope) == 6, str(sorted(scope)))

    unscoped_documents = [doc for doc, names in scope.items() if not names]
    check("every corpus document declares a signal scope",
          not unscoped_documents, str(unscoped_documents))

    relevant_problems: list[str] = []
    irrelevant_problems: list[str] = []
    fired_by_case: dict[str, tuple[str, ...]] = {}

    for case in CASES:
        report = case.report()
        if report is None:
            skip(f"{case.case_id}: capture unavailable", "the pcap is not present")
            continue
        fired = extract_signals(report).types()
        fired_by_case[case.case_id] = fired

        for document in sorted(case.relevant_documents):
            verdict = assess(scope[document], fired)
            if verdict.compatibility is Compatibility.UNDECLARED:
                relevant_problems.append(f"{case.case_id}:{document}")

        for document in sorted(case.irrelevant_documents):
            verdict = assess(scope[document], fired)
            if verdict.compatibility is not Compatibility.UNDECLARED:
                irrelevant_problems.append(f"{case.case_id}:{document}")

    check("no document a case calls relevant is demoted by the rule",
          not relevant_problems, str(relevant_problems))
    check("every document a case calls irrelevant is demoted by the rule",
          not irrelevant_problems, str(irrelevant_problems))

    # The four cases step 8 measured as wrong.  Named individually, because a
    # regression in one of them is the thing this whole step exists to prevent.
    for case_id, document in (("normal-cdn-multi-host", "dns-tunneling"),
                              ("unknown-application", "dns-tunneling"),
                              ("real-capture-benign", "dns-tunneling"),
                              ("knowledge-conflicts-observation", "dns-tunneling"),
                              ("knowledge-conflicts-observation",
                               "suspicious-dns-indicators")):
        fired = fired_by_case.get(case_id)
        if fired is None:
            skip(f"{case_id}: {document} is demoted", "capture unavailable")
            continue
        verdict = assess(scope[document], fired)
        check(f"{case_id}: {document} is demoted",
              verdict.compatibility is Compatibility.UNDECLARED, verdict.note)

    # And the recall miss, from the other direction.
    fired = fired_by_case.get("normal-cdn-multi-host")
    if fired is not None:
        verdict = assess(scope["dns-normal-behaviour"], fired)
        check("normal-cdn-multi-host: dns-normal-behaviour is promoted, not demoted",
              verdict.compatibility is Compatibility.DECLARED, verdict.note)

    check("a document declaring a signal no case fires is still classifiable",
          assess(scope["dns-tunneling"], ("scan_half_open",)).compatibility
          is Compatibility.UNDECLARED)


# ===========================================================================
# C. Ranking mechanics
# ===========================================================================
def _fixture() -> tuple[VectorStore, EmbeddingModel, CaptureReport]:
    """Two documents, arranged to produce the failure mode under test.

    The capture fires ``baseline_web_browsing`` and nothing else.

    * ``lure-notes`` is written in the vocabulary the browsing query uses -- the
      word "browsing", over and over -- so the keyword encoder scores it highest
      for every query.  Its ``applies_to`` declares ``dns_high_volume``: it is
      topically closest and situationally wrong, which is exactly the shape of
      the false positives step 8 measured.
    * ``fit-notes`` declares ``baseline_web_browsing`` and is written about
      ports, so it scores *lower* on similarity alone.

    Similarity alone therefore ranks the wrong document first.  That is the
    point of the fixture: a policy that only reordered results that were
    already correct would prove nothing.
    """
    embedder = EmbeddingModel(EmbeddingConfig(model_name=MODEL_NAME),
                              encoder=KeywordEncoder())
    chunks = []
    chunks.extend(chunk_document(parse_document(
        document_text("lure-notes", "Lure Notes", ["dns_high_volume"], "browsing"),
        "protocols/lure-notes.md")))
    chunks.extend(chunk_document(parse_document(
        document_text("fit-notes", "Fit Notes", ["baseline_web_browsing"], "port"),
        "protocols/fit-notes.md")))
    store = build_store(tuple(chunks), embedder)
    return store, embedder, capture([flow(i) for i in range(6)])


def test_ranking() -> None:
    print("\nC. Ranking mechanics")

    store, embedder, report = _fixture()
    signals = extract_signals(report)
    check("the fixture capture fires only the browsing signal",
          signals.types() == ("baseline_web_browsing",), str(signals.types()))

    check("compatibility ranking is off by default, on measured evidence",
          RetrievalConfig().affinity is AffinityMode.OFF)

    off = RetrievalConfig(affinity=AffinityMode.OFF, max_per_document=None,
                          final_top_k=99, per_query_top_k=99)
    on = RetrievalConfig(affinity=AffinityMode.RANK, max_per_document=None,
                         final_top_k=99, per_query_top_k=99)
    plain = retrieve_for_signals(signals, store, embedder, off)
    tiered = retrieve_for_signals(signals, store, embedder, on)

    check("without compatibility the wrong document ranks first",
          plain.chunks[0].document_id == "lure-notes",
          plain.chunks[0].document_id)
    check("with compatibility the compatible document ranks first",
          tiered.chunks[0].document_id == "fit-notes",
          tiered.chunks[0].document_id)

    check("nothing is removed -- only reordered",
          set(plain.chunk_ids()) == set(tiered.chunk_ids()),
          str(set(plain.chunk_ids()) ^ set(tiered.chunk_ids())))

    plain_scores = {c.chunk_id: c.similarity for c in plain.chunks}
    tiered_scores = {c.chunk_id: c.similarity for c in tiered.chunks}
    check("no cosine score is altered by compatibility",
          plain_scores == tiered_scores)

    demoted = [c for c in tiered.chunks
               if c.compatibility is Compatibility.UNDECLARED]
    check("the demoted chunks are the ones from the incompatible document",
          demoted and all(c.document_id == "lure-notes" for c in demoted),
          str([c.document_id for c in demoted]))
    check("every demoted result says why it was demoted",
          all("none of which" in c.affinity_note for c in demoted),
          str([c.affinity_note for c in demoted][:1]))
    check("every promoted result says why it was promoted",
          all("which this capture produced" in c.affinity_note
              for c in tiered.chunks
              if c.compatibility is Compatibility.DECLARED))

    check("compatible chunks all precede incompatible ones",
          [c.affinity_tier for c in tiered.chunks]
          == sorted(c.affinity_tier for c in tiered.chunks))
    compatible = [c for c in tiered.chunks
                  if c.compatibility is Compatibility.DECLARED]
    check("cosine order is untouched inside a tier",
          [c.similarity for c in compatible]
          == sorted((c.similarity for c in compatible), reverse=True))

    check("affinity=off reproduces similarity-only ranking exactly",
          [c.chunk_id for c in plain.chunks]
          == [c.chunk_id for c in retrieve_for_signals(signals, store, embedder,
                                                       off).chunks])
    check("the parameters record which policy produced the result",
          tiered.parameters["affinity"] == "rank"
          and plain.parameters["affinity"] == "off")
    check("a demotion is disclosed in the report notes",
          any("ranked after the compatible ones" in note for note in tiered.notes),
          str(tiered.notes))

    again = retrieve_for_signals(signals, store, embedder, on)
    check("tiered retrieval is byte-identical between runs",
          again.to_json(include_text=True) == tiered.to_json(include_text=True))

    capped = retrieve_for_signals(
        signals, store, embedder,
        RetrievalConfig(affinity=AffinityMode.RANK, max_per_document=1,
                        final_top_k=8, per_query_top_k=99))
    documents = [c.document_id for c in capped.chunks]
    check("max_per_document=1 gives one chunk per document",
          len(documents) == len(set(documents)), str(documents))
    check("and the compatible document is the one that keeps its slot",
          documents[0] == "fit-notes", str(documents))


# ===========================================================================
# D. Query framing
# ===========================================================================
def test_query_framing() -> None:
    print("\nD. Query framing")

    report = extract_signals(capture([flow(i) for i in range(6)]))
    topical = build_queries(report, RetrievalConfig(query_style="topical"))
    security = build_queries(report, RetrievalConfig(query_style="security"))

    # Step 9 shipped "topical"; step 10 measured it and put "security" back.
    # The assertion tracks the shipped wording, so a future change to the
    # default has to be a deliberate edit here rather than a silent drift.
    check("the shipped style is the one every measurement was taken with",
          RetrievalConfig().query_style == DEFAULT_QUERY_STYLE == "security")
    check("the neutral wording is still selectable as a candidate",
          RetrievalConfig(query_style="topical").lead_in == QUERY_LEAD_IN["topical"])
    check("both styles build the same number of queries",
          len(topical) == len(security) and len(topical) > 0)
    check("the styles differ only in the opening words",
          all(t.text[len(QUERY_LEAD_IN["topical"]):]
              == s.text[len(QUERY_LEAD_IN["security"]):]
              for t, s in zip(topical, security)))
    check("the neutral lead-in does not name a stance",
          all(not q.text.lower().startswith("network security") for q in topical),
          topical[0].text.splitlines()[0])
    check("the original wording is preserved exactly, for comparison",
          all(q.text.startswith("Network security knowledge about: ")
              for q in security))
    check("every query still carries the bge prefix exactly once",
          all(q.embedding_text.count("Represent this sentence for searching") == 1
              for q in topical + security))
    check("no query carries a hostname or an address",
          all("example" not in q.text and "." not in q.text.split(":", 1)[0]
              for q in topical))

    raises("an unknown query style is refused rather than silently defaulted",
           ValueError, lambda: RetrievalConfig(query_style="salesy"))
    check("the lead-in table has an entry for every style",
          set(QUERY_LEAD_IN) == {"topical", "security"})


# ===========================================================================
# E. Request-size guard
# ===========================================================================
def test_request_guard() -> None:
    print("\nE. Request-size guard")

    check("the estimate grows with content",
          estimate_request_tokens([{"role": "user", "content": "x" * 3500}])
          > estimate_request_tokens([{"role": "user", "content": "x" * 350}]))
    check("an empty message list estimates zero",
          estimate_request_tokens([]) == 0)
    check("the estimate counts every message",
          estimate_request_tokens([{"role": "user", "content": "x" * 350}] * 2)
          > estimate_request_tokens([{"role": "user", "content": "x" * 350}]))

    groq = get_provider_spec(Provider.GROQ)
    check("no provider ships a guessed request ceiling",
          all(get_provider_spec(p).max_input_tokens is None for p in Provider),
          str({p.value: get_provider_spec(p).max_input_tokens for p in Provider}))
    check("groq still says what to do about an oversized request",
          len(groq.oversize_hint) > 20)

    check("with no ceiling configured the default config has none",
          AIConfig(provider=Provider.GROQ).max_input_tokens is None)
    config = AIConfig(provider=Provider.GROQ, max_input_tokens=4000)
    check("an explicit ceiling is honoured",
          config.max_input_tokens == 4000)
    check("a ceiling of zero means no ceiling",
          AIConfig(provider=Provider.GROQ, max_input_tokens=0).max_input_tokens is None)

    from ai.llm_client import _oversize_detail

    small = [{"role": "user", "content": "x" * 100}]
    huge = [{"role": "user", "content": "x" * 200_000}]
    check("an ordinary request is not refused",
          _oversize_detail(small, config) is None)
    detail = _oversize_detail(huge, config)
    check("an oversized request is refused before it is sent", detail is not None)
    check("the refusal says how big it was and what the limit is",
          detail is not None and str(config.max_input_tokens) in detail
          and "not sent" in detail, str(detail))
    check("the refusal names something the user can change",
          detail is not None and "--rag-max-items" in detail)
    check("nothing is refused when no ceiling is configured",
          _oversize_detail(huge, AIConfig(provider=Provider.GROQ)) is None)
    check("and the shipped default configures none, so nothing changes by default",
          _oversize_detail(huge, AIConfig(provider=Provider.OLLAMA)) is None)

    check("the failure reason exists and is distinct",
          FailureReason.REQUEST_TOO_LARGE.value == "request_too_large")

    # The path that needs no guess: the provider said 413, so we know.
    from ai.llm_client import OpenAIClient
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a base requirement
        skip("an HTTP 413 is classified as too-large, not as a transient error",
             "the openai package is not installed")
    else:
        class _Response:
            def __init__(self, status: int) -> None:
                self.status_code = status
                self.headers: dict[str, str] = {}
                self.request = None

        def status_error(status: int) -> Exception:
            error = openai.APIStatusError.__new__(openai.APIStatusError)
            error.status_code = status
            error.response = _Response(status)
            error.body = None
            error.message = "Request too large"
            return error

        reason, retry = OpenAIClient._classify(status_error(413))
        check("an HTTP 413 is classified as too-large, not as a transient error",
              reason is FailureReason.REQUEST_TOO_LARGE, str(reason))
        check("and is not retried, because the request will be the same size",
              retry is False)
        other, other_retry = OpenAIClient._classify(status_error(503))
        check("other status errors keep their existing classification",
              other is FailureReason.API_ERROR and other_retry is True)
    guidance = explain_failure(FailureReason.REQUEST_TOO_LARGE)
    check("and comes with guidance rather than a bare code",
          len(guidance) > 40 and ("smaller" in guidance or "less" in guidance),
          guidance)
    with_provider = explain_failure(FailureReason.REQUEST_TOO_LARGE,
                                    AIConfig(provider=Provider.GROQ))
    check("guidance names the knob that makes the request smaller",
          "--rag-max-items" in with_provider and "Groq" in with_provider,
          with_provider)

    # The guard reports sizes, never content: a message body that happened to
    # contain a key must not travel into the failure detail.
    secret = [{"role": "user", "content": "gsk_" + "a" * 200_000}]
    leaked = _oversize_detail(secret, config) or ""
    check("the refusal quotes no part of the request", "gsk_" not in leaked)


# ===========================================================================
# F. End to end, on the real corpus, with no model
# ===========================================================================
def test_corpus_shapes() -> None:
    print("\nF. Real corpus, no embedding model")

    chunks = chunk_corpus(load_corpus())
    check("the corpus still chunks", len(chunks) > 20, str(len(chunks)))
    check("every chunk carries the document's declared scope",
          all(isinstance(chunk.applies_to, list) for chunk in chunks))

    scoped = {chunk.document_id: tuple(chunk.applies_to) for chunk in chunks}
    check("a document's scope is identical on all of its chunks",
          all(tuple(chunk.applies_to) == scoped[chunk.document_id]
              for chunk in chunks))

    digest = hashlib.sha256(
        "\n".join(sorted(chunk.chunk_id for chunk in chunks)).encode("utf-8")
    ).hexdigest()
    again = hashlib.sha256(
        "\n".join(sorted(chunk.chunk_id
                         for chunk in chunk_corpus(load_corpus()))).encode("utf-8")
    ).hexdigest()
    check("chunk ids are stable across two loads", digest == again)


# ===========================================================================
# G. The accept/reject rule
# ===========================================================================
def _variant(recall: float, precision: float, irrelevant: int = 0,
             missed: int = 0) -> dict:
    """A minimal stand-in for one row of the before/after table.

    Only the headline block matters to the rule; the rest of a real row is
    display detail.
    """
    return {"headline": {"recall": recall, "precision": precision,
                         "irrelevant": irrelevant, "missed": missed}}


def test_decision_rule() -> None:
    print("\nG. The accept/reject rule")

    from run_rag_evaluation import _decide

    better = _decide(_variant(0.94, 0.71, irrelevant=3),
                     _variant(0.94, 0.85, irrelevant=0))
    check("precision up, recall level, fewer irrelevant hits -> accept",
          better["verdict"] == "accept", str(better))
    check("and the verdict shows the numbers behind it",
          any("supplied recall" in line for line in better["evidence"])
          and any("supplied precision" in line for line in better["evidence"]))

    traded = _decide(_variant(0.94, 0.71, irrelevant=3),
                     _variant(1.00, 0.55, irrelevant=5))
    check("recall bought with precision -> reject",
          traded["verdict"] == "reject", str(traded))

    collapsed = _decide(_variant(0.94, 0.71, irrelevant=3),
                        _variant(0.60, 0.99, irrelevant=0))
    check("precision bought with a material recall loss -> reject",
          collapsed["verdict"] == "reject", str(collapsed))

    marginal = _decide(_variant(0.94, 0.71, irrelevant=3),
                       _variant(0.92, 0.85, irrelevant=0))
    check("a recall loss under the stated margin is not fatal on its own",
          marginal["verdict"] == "accept", str(marginal))

    same = _decide(_variant(0.94, 0.71, irrelevant=3),
                   _variant(0.94, 0.71, irrelevant=3))
    check("no measurable change -> reject, not accept by default",
          same["verdict"] == "reject", str(same))

    more_bad = _decide(_variant(0.94, 0.71, irrelevant=1),
                       _variant(0.99, 0.75, irrelevant=4))
    check("more irrelevant documents retrieved -> reject even if both metrics rise",
          more_bad["verdict"] == "reject", str(more_bad))

    blank = _variant(0.94, 0.71)
    blank["headline"]["recall"] = None
    check("an undefined metric is reported undecided, never as an improvement",
          _decide(_variant(0.94, 0.71), blank)["verdict"] == "undecided")
    check("the rule judges the supplied set, not the retrieved one",
          "supplied" in " ".join(better["evidence"]))


# ===========================================================================
# H. The shipped defaults, pinned
# ===========================================================================
#: Every production default this project has deliberately chosen, in one place.
#:
#: The point is not that these values are right -- it is that changing one
#: should be a decision someone makes on purpose, in this dictionary, with the
#: measurement that justified it. Three times now a default has moved on the
#: strength of an argument and been moved back by the first real measurement,
#: so the defaults are pinned and the pin is the paperwork.
#:
#: To adopt a measured candidate: change the value here and in the module that
#: owns it, in the same commit, and say in the message which evaluation run
#: supports it.
SHIPPED_DEFAULTS: dict[str, object] = {
    # -- retrieval (ai/rag/retrieval.py) --------------------------------
    "per_query_top_k": 4,
    "final_top_k": 8,
    "max_per_document": 2,
    "min_similarity": None,
    "affinity": AffinityMode.OFF,
    "query_style": "security",
    "include_capture_query": True,
    # -- context budget (ai/rag/context.py) ------------------------------
    "max_items": 4,
    "max_chars": 3000,
    "max_total_tokens": 900,
    # -- provider and capture (ai/config.py, ai/providers.py) ------------
    "max_input_tokens": None,
    "max_flows": 40,
}


def test_shipped_defaults() -> None:
    print("\nH. The shipped defaults are exactly what was decided")

    from ai.config import DEFAULT_MAX_FLOWS, AIConfig
    from ai.rag.context import (
        DEFAULT_MAX_CHARS,
        DEFAULT_MAX_ITEMS,
        DEFAULT_MAX_TOKENS,
        KnowledgeContextConfig,
    )

    retrieval = RetrievalConfig()
    budget = KnowledgeContextConfig()
    expected = SHIPPED_DEFAULTS

    for name, actual in (
        ("per_query_top_k", retrieval.per_query_top_k),
        ("final_top_k", retrieval.final_top_k),
        ("max_per_document", retrieval.max_per_document),
        ("min_similarity", retrieval.min_similarity),
        ("affinity", retrieval.affinity),
        ("query_style", retrieval.query_style),
        ("include_capture_query", retrieval.include_capture_query),
        ("max_items", budget.max_items),
        ("max_chars", budget.max_chars),
        ("max_total_tokens", budget.max_total_tokens),
        ("max_input_tokens", AIConfig().max_input_tokens),
        ("max_flows", DEFAULT_MAX_FLOWS),
    ):
        check(f"shipped {name} is {expected[name]!r}",
              actual == expected[name], f"found {actual!r}")

    # The module constants and the dataclass defaults must not drift apart:
    # a caller who reads DEFAULT_MAX_CHARS and a caller who builds a bare
    # KnowledgeContextConfig have to get the same budget.
    check("the budget constants match the budget dataclass",
          (DEFAULT_MAX_ITEMS, DEFAULT_MAX_CHARS, DEFAULT_MAX_TOKENS)
          == (budget.max_items, budget.max_chars, budget.max_total_tokens))

    # -- overrides still work, which is what makes pinning safe ----------
    override = RetrievalConfig(per_query_top_k=6)
    check("a caller can still override per_query_top_k",
          override.per_query_top_k == 6 and RetrievalConfig().per_query_top_k
          == expected["per_query_top_k"])
    custom = KnowledgeContextConfig(max_chars=4000, max_total_tokens=1200)
    check("a caller can still override the budget",
          (custom.max_chars, custom.max_total_tokens) == (4000, 1200))
    check("and the shipped budget is unchanged by that",
          (KnowledgeContextConfig().max_chars,
           KnowledgeContextConfig().max_total_tokens)
          == (expected["max_chars"], expected["max_total_tokens"]))

    env = KnowledgeContextConfig.from_env
    check("the environment override mechanism is still present", callable(env))


def test_candidate_definitions_match_shipped() -> None:
    print("\nH2. The evaluation's baseline is the configuration that ships")

    from evaluation.candidates import by_name

    baseline = by_name("baseline")
    for key, value in dict(baseline.retrieval).items():
        check(f"baseline candidate {key} matches the shipped default",
              SHIPPED_DEFAULTS[key] == value, f"{SHIPPED_DEFAULTS[key]!r} vs {value!r}")
    for key, value in dict(baseline.budget).items():
        check(f"baseline candidate {key} matches the shipped default",
              SHIPPED_DEFAULTS[key] == value, f"{SHIPPED_DEFAULTS[key]!r} vs {value!r}")

    # D-partial is the three-change set step 11 proposed. It is a candidate,
    # not a default, and it must differ from the shipped configuration in
    # exactly those three parameters -- no more, or the comparison stops
    # isolating what it claims to isolate.
    partial = by_name("D-partial")
    merged = dict(partial.retrieval) | dict(partial.budget)
    differs = {key for key, value in merged.items() if SHIPPED_DEFAULTS[key] != value}
    check("D-partial differs from the shipped defaults in exactly three parameters",
          differs == {"per_query_top_k", "max_chars", "max_total_tokens"},
          str(sorted(differs)))

    full = by_name("D")
    merged_full = dict(full.retrieval) | dict(full.budget)
    differs_full = {key for key, value in merged_full.items()
                    if SHIPPED_DEFAULTS[key] != value}
    check("candidate D differs in six, which is why it is not those three",
          differs_full == {"per_query_top_k", "max_per_document", "min_similarity",
                           "max_items", "max_chars", "max_total_tokens"},
          str(sorted(differs_full)))
    check("so D-partial and D are genuinely different configurations",
          merged != merged_full)


# ===========================================================================
# I. The provider request path -- the 413 and the 400
# ===========================================================================
def real_capture():
    """The real pcap through the unmodified DPI engine, or None."""
    import io as _io
    import os as _os
    from contextlib import redirect_stdout
    from pathlib import Path

    if not Path("test_dpi.pcap").is_file():
        return None
    from ai.extractor import build_capture_report
    from dpi.dpi_engine import Config, DPIEngine

    with redirect_stdout(_io.StringIO()):
        engine = DPIEngine(Config())
        engine.initialize()
        engine.process_file("test_dpi.pcap", _os.devnull)
        snapshot = engine.get_flow_snapshot()
    return build_capture_report(snapshot, "test_dpi.pcap", AIConfig())


def test_capture_rendering() -> None:
    print("\nI. Capture rendering: smaller request, same information")

    from ai.prompts import (
        CAPTURE_FORMATS,
        DEFAULT_CAPTURE_FORMAT,
        build_messages,
        prompt_version,
        render_capture,
    )

    report = capture([flow(i) for i in range(6)]
                     + [flow(6, server_name=None, application="Unknown")])
    payload = report.model_dump(mode="json", exclude_none=True)

    table = render_capture(report, "table")
    as_json = render_capture(report, "json")

    check("the table layout is the shipped default",
          DEFAULT_CAPTURE_FORMAT == "table" and AIConfig().capture_format == "table")
    check("the json layout is still available",
          set(CAPTURE_FORMATS) == {"table", "json"})
    check("the json layout reproduces the pre-2.0 body byte for byte",
          as_json == json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))

    # -- nothing is lost -------------------------------------------------
    rows = table.split("===== BEGIN FLOWS =====\n")[1].split("\n===== END FLOWS")[0]
    lines = rows.splitlines()
    columns = lines[0].split("|")
    check("every flow gets a row",
          len(lines) - 1 == len(payload["flows"]), f"{len(lines) - 1} rows")
    check("every field of every flow gets a column",
          set(columns) == {name for f in payload["flows"] for name in f},
          str(sorted(set(columns))))

    recovered = []
    for line in lines[1:]:
        cells = line.split("|")
        recovered.append({name: cell for name, cell in zip(columns, cells)})
    intact = True
    for original, back in zip(payload["flows"], recovered):
        for name in columns:
            value = original.get(name)
            expected = ("" if value is None else
                        "1" if value is True else "0" if value is False else str(value))
            if back[name] != expected:
                intact = False
                check(f"flow field {name} survives the rendering", False,
                      f"{expected!r} became {back[name]!r}")
    check("every value survives the rendering unchanged", intact)

    check("capture-wide fields are still present",
          all(key in table for key in ("totals", "application_distribution",
                                       "redaction_mode", "capture_name")))
    check("the block explains its own layout",
          "first line names the columns" in table and "1 (true) and 0 (false)" in table)

    # -- and it is materially smaller -------------------------------------
    check("the table body is smaller than the json body",
          len(table) < len(as_json), f"{len(table)} vs {len(as_json)}")

    live = real_capture()
    if live is None:
        skip("the real capture shrinks materially", "test_dpi.pcap is not present")
    else:
        big = estimate_request_tokens(build_messages(live, None, None, "json"))
        small = estimate_request_tokens(build_messages(live, None, None, "table"))
        check("the real capture request shrinks by at least a third",
              small <= big * 0.67, f"{big} -> {small} tokens")
        check("and the saving comes from layout, not from dropped flows",
              str(len(live.flows)) in render_capture(live, "table"))

    # -- determinism -------------------------------------------------------
    check("rendering is byte-identical between runs",
          render_capture(report, "table") == table)
    check("the recorded prompt version distinguishes the two layouts",
          prompt_version(False, "table") != prompt_version(False, "json")
          and prompt_version(False, "json").endswith("+json"))

    raises("an unknown capture format is refused", ValueError,
           lambda: render_capture(report, "yaml"))


def test_capture_rendering_safety() -> None:
    print("\nI2. The table layout cannot be broken by capture content")

    from ai.prompts import render_capture

    hostile = capture([flow(0, server_name="a|b.example.com"),
                       flow(1, server_name="back\\slash.example.com")])
    table = render_capture(hostile, "table")
    rows = table.split("===== BEGIN FLOWS =====\n")[1].split("\n===== END FLOWS")[0]
    columns = rows.splitlines()[0].split("|")
    for line in rows.splitlines()[1:]:
        # An escaped separator must not create a column.
        unescaped = re.sub(r"\\.", "X", line)
        check("a hostname containing the separator does not invent a column",
              len(unescaped.split("|")) == len(columns),
              f"{len(unescaped.split('|'))} cells for {len(columns)} columns")
    check("the separator is escaped rather than dropped", "\\|" in table)

    check("no address appears that the json layout would not also carry",
          set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", table))
          <= set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                            render_capture(hostile, "json"))))
    check("the redaction mode still travels with the data",
          "redact_private" in table)


def test_structured_output_400() -> None:
    print("\nI3. The structured-output 400 (json_validate_failed)")

    from ai.providers import (
        PROVIDER_OMITTED_FIELDS,
        StructuredMode,
        get_provider_spec,
        to_strict_json_schema,
        unenforced_constants,
    )
    from ai.schemas import AnalysisResult

    raw = AnalysisResult.model_json_schema()

    # The reproduction: the schema as it was sent when the live call failed.
    before = to_strict_json_schema(raw, omit=())
    check("the failing schema required a const the model had to guess",
          "properties.schema_version" in unenforced_constants(before)
          and "schema_version" in before["required"],
          str(unenforced_constants(before)))

    after = to_strict_json_schema(raw)
    check("the shipped schema carries no const at all",
          unenforced_constants(after) == [], str(unenforced_constants(after)))
    check("schema_version is withheld from the provider",
          PROVIDER_OMITTED_FIELDS == ("schema_version",)
          and "schema_version" not in after["properties"])
    check("and is not left required, which nothing could satisfy",
          "schema_version" not in after["required"])

    # Everything else the analyst actually produces is still demanded.
    for name in ("summary", "observed_facts", "interpretation", "uncertainties",
                 "traffic_type", "risk_level", "risk_rationale", "confidence",
                 "indicators", "recommended_actions", "notable_flow_ids",
                 "knowledge_refs"):
        check(f"the provider still requires {name}", name in after["required"])
    check("strict mode still forbids extra properties",
          after["additionalProperties"] is False)

    # A response shaped like the corrected schema validates, and our own
    # Literal is restored by the default rather than by the model.
    body = {
        "summary": "Six flows of ordinary web traffic.",
        "observed_facts": ["6 flows target port 443."],
        "interpretation": ["Consistent with browsing."],
        "uncertainties": ["Payloads are encrypted."],
        "traffic_type": "web_browsing", "risk_level": "informational",
        "risk_rationale": "Well-known destinations.", "confidence": 0.6,
        "indicators": [], "recommended_actions": [], "notable_flow_ids": [0],
        "knowledge_refs": ["K1"],
    }
    parsed = AnalysisResult.model_validate_json(json.dumps(body))
    check("a response without schema_version validates",
          parsed.schema_version == "1.1")
    check("and the Literal is still enforced when the field IS supplied",
          not _accepts(AnalysisResult, dict(body, schema_version="1.0")))
    check("citation validation is untouched",
          parsed.knowledge_refs == ["K1"]
          and AnalysisResult.validate_knowledge_references(parsed, ("K1",)) == []
          and AnalysisResult.validate_knowledge_references(parsed, ()) != [])

    check("only the JSON_SCHEMA and NATIVE_PARSE providers are affected",
          get_provider_spec(Provider.OLLAMA).structured_mode
          is StructuredMode.JSON_OBJECT)


def _accepts(model, payload) -> bool:
    try:
        model.model_validate(payload)
    except Exception:  # noqa: BLE001 - the point is that it rejects
        return False
    return True


def test_413_regression() -> None:
    print("\nI4. HTTP 413 stays classified, unretried and quiet about secrets")

    import openai

    from ai.llm_client import OpenAIClient, _oversize_detail

    class _Response:
        def __init__(self, status: int) -> None:
            self.status_code = status
            self.headers: dict[str, str] = {}
            self.request = None

    def status_error(status: int, message: str = "Request too large") -> Exception:
        error = openai.APIStatusError.__new__(openai.APIStatusError)
        error.status_code = status
        error.response = _Response(status)
        error.body = None
        error.message = message
        return error

    reason, retry = OpenAIClient._classify(status_error(413))
    check("413 is REQUEST_TOO_LARGE",
          reason is FailureReason.REQUEST_TOO_LARGE, str(reason))
    check("413 is not retried", retry is False)

    rate = OpenAIClient._classify(openai.RateLimitError.__new__(openai.RateLimitError))
    check("429 is still RATE_LIMITED and still retried",
          rate == (FailureReason.RATE_LIMITED, True), str(rate))

    guidance = explain_failure(FailureReason.REQUEST_TOO_LARGE,
                              AIConfig(provider=Provider.GROQ))
    check("the guidance names the knobs that make a request smaller",
          "--max-flows" in guidance and "--rag-max-items" in guidance)

    FAKE = "gsk_" + "Q" * 44
    leaky = _oversize_detail([{"role": "user", "content": FAKE * 4000}],
                             AIConfig(provider=Provider.GROQ, max_input_tokens=100),
                             None) or ""
    check("an oversize refusal quotes no part of the request", FAKE not in leaky)
    from ai.llm_client import describe_provider_error
    described = describe_provider_error(status_error(413, f"key {FAKE} too large"))
    check("a 413 diagnostic carries no key", FAKE not in described, described[:80])
    check("but still carries the status and a request id slot",
          "HTTP 413" in described, described[:80])


# ===========================================================================
def main() -> int:
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print("RAG step 9 -- retrieval quality and production hardening")

    test_policy()
    test_corpus_agreement()
    test_ranking()
    test_query_framing()
    test_request_guard()
    test_corpus_shapes()
    test_decision_rule()
    test_shipped_defaults()
    test_candidate_definitions_match_shipped()
    test_capture_rendering()
    test_capture_rendering_safety()
    test_structured_output_400()
    test_413_regression()

    total = _passed + _failed
    suffix = f", {_skipped} skipped" if _skipped else ""
    print(f"\n{_passed}/{total} checks passed{suffix}")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
