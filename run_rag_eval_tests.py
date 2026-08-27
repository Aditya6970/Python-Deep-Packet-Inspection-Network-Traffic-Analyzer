"""Tests for the evaluation harness itself -- RAG step 8.

Why the harness needs its own tests
-----------------------------------
An evaluation harness is measuring equipment, and equipment that is wrong is
worse than none: it produces numbers that look like evidence. These checks
cover the arithmetic against worked examples, the dataset's internal
consistency, the sweep accounting, and the harness's own privacy and
determinism.

They do **not** test whether retrieval is any good. That is what
``run_rag_evaluation.py`` measures, and a test that asserted a particular
recall would be pinning a quality result into the test suite, where a real
regression and a deliberate improvement would look identical.

The sweep machinery is exercised through a keyword encoder defined here, so the
plumbing is verified without a model download. The numbers it produces are
meaningless by construction and nothing here asserts anything about them --
only that the accounting adds up.

Run::

    python run_rag_eval_tests.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Import the package so its console-encoding fix is applied before any output.
import dpi  # noqa: F401
import run_rag_evaluation as harness
from ai.rag.context import KnowledgeContextConfig
from ai.rag.embeddings import EmbeddingConfig, EmbeddingModel
from ai.rag.retrieval import RetrievalConfig
from ai.rag.signals import extract_signals
from ai.rag.vector_store import VectorRecord, VectorStore
from ai.schemas import AnalysisResult
from evaluation.cases import CASES, CORPUS_DOCUMENT_IDS, EvaluationCase, cases_for_live
from evaluation.metrics import (
    MetricSummary,
    RetrievalMetrics,
    aggregate,
    deduplicate,
    hit_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    score_ranking,
)

_passed = 0
_failed = 0
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def check(label: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}" + (f"  -- {detail}" if detail else ""))


def raises(label: str, expected: type[Exception], call) -> None:
    try:
        call()
    except expected as exc:
        check(label, len(str(exc)) > 5, f"error message too terse: {str(exc)!r}")
    except Exception as exc:  # noqa: BLE001 - wrong type is the failure
        check(label, False, f"raised {type(exc).__name__}: {exc}")
    else:
        check(label, False, f"no {expected.__name__} was raised")


def near(value: float | None, expected: float) -> bool:
    return value is not None and abs(value - expected) < 1e-9


# ===========================================================================
# 1. Metric arithmetic, against worked examples
# ===========================================================================
def test_metrics() -> None:
    print("\n1. Metric arithmetic")

    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d", "e"}

    check("hit@1 is 0 when the top result is irrelevant",
          near(hit_at_k(ranked, relevant, 1), 0.0))
    check("hit@2 is 1 once a relevant id appears",
          near(hit_at_k(ranked, relevant, 2), 1.0))
    check("recall@2 counts one of three relevant",
          near(recall_at_k(ranked, relevant, 2), 1 / 3))
    check("recall@4 counts two of three relevant",
          near(recall_at_k(ranked, relevant, 4), 2 / 3))
    check("precision@2 counts one of two returned",
          near(precision_at_k(ranked, relevant, 2), 0.5))
    check("precision@4 counts two of four returned",
          near(precision_at_k(ranked, relevant, 4), 0.5))
    check("reciprocal rank is 1/2 for a hit at position two",
          near(mean_reciprocal_rank(ranked, relevant), 0.5))
    check("reciprocal rank is 1.0 for a hit at position one",
          near(mean_reciprocal_rank(["b", "a"], relevant), 1.0))
    check("reciprocal rank is 0.0 when nothing relevant appears",
          near(mean_reciprocal_rank(["x", "y"], relevant), 0.0))
    check("recall never exceeds 1.0",
          near(recall_at_k(["b", "d", "e"], relevant, 8), 1.0))

    # -- K edges ----------------------------------------------------------
    check("K=0 finds no hit", near(hit_at_k(ranked, relevant, 0), 0.0))
    check("K=0 has zero recall", near(recall_at_k(ranked, relevant, 0), 0.0))
    check("K=0 has undefined precision -- nothing was considered",
          precision_at_k(ranked, relevant, 0) is None)
    check("K larger than the result count uses what exists",
          near(recall_at_k(ranked, relevant, 99), 2 / 3))
    check("precision divides by results returned, not by K",
          near(precision_at_k(["b"], relevant, 5), 1.0),
          "one perfect result at K=5 is precision 1.0, not 0.2")
    raises("a negative K is rejected", ValueError,
           lambda: recall_at_k(ranked, relevant, -1))

    # -- no relevance labels ----------------------------------------------
    check("hit is undefined with no relevant ids", hit_at_k(ranked, set(), 3) is None)
    check("recall is undefined with no relevant ids",
          recall_at_k(ranked, set(), 3) is None)
    check("precision is undefined with no relevant ids",
          precision_at_k(ranked, set(), 3) is None)
    check("MRR is undefined with no relevant ids",
          mean_reciprocal_rank(ranked, set()) is None)
    check("undefined is None, never zero",
          recall_at_k(ranked, set(), 3) is None
          and recall_at_k(ranked, {"zzz"}, 3) == 0.0,
          "an unmet relevance is 0.0; an absent one is None")

    # -- empty ranking ----------------------------------------------------
    check("an empty ranking finds no hit", near(hit_at_k([], relevant, 5), 0.0))
    check("an empty ranking has zero recall", near(recall_at_k([], relevant, 5), 0.0))
    check("an empty ranking has undefined precision",
          precision_at_k([], relevant, 5) is None)
    check("an empty ranking has zero reciprocal rank",
          near(mean_reciprocal_rank([], relevant), 0.0))

    # -- duplicates -------------------------------------------------------
    check("deduplicate keeps first occurrence in order",
          deduplicate(["b", "a", "b", "c", "a"]) == ["b", "a", "c"])
    check("duplicate ids collapse before the cut-off",
          near(recall_at_k(["a", "a", "a", "b"], relevant, 2), 1 / 3),
          "three copies of 'a' must not crowd 'b' out of the top 2")
    check("without dedupe the duplicates do fill the window",
          near(recall_at_k(["a", "a", "a", "b"], relevant, 2, dedupe=False), 0.0))
    check("duplicates can be kept when the level demands it",
          near(precision_at_k(["a", "a"], {"a"}, 2, dedupe=False), 1.0))
    check("duplicate relevant ids are a set, so they count once",
          near(recall_at_k(["b"], ["b", "b", "d"], 5), 0.5))

    # -- score_ranking ----------------------------------------------------
    scored = score_ranking(ranked, relevant, 2, "document", irrelevant={"a"})
    check("score_ranking reports what it considered", scored.considered == 2)
    check("score_ranking reports what was returned", scored.returned == 4)
    check("score_ranking counts relevant found", scored.relevant_found == 1)
    check("score_ranking counts labelled irrelevant found", scored.irrelevant_found == 1)
    check("score_ranking carries the level", scored.level == "document")
    check("score_ranking renders a row", "K=2" in scored.row())
    check("an unjudged id counts as neither",
          score_ranking(["z"], relevant, 1, "document",
                        irrelevant={"a"}).irrelevant_found == 0)
    check("metrics are immutable", RetrievalMetrics.model_config["frozen"] is True)
    raises("an inconsistent metric row is rejected", ValueError,
           lambda: RetrievalMetrics(k=1, level="document", returned=1, considered=5,
                                    relevant_total=1, relevant_found=0,
                                    irrelevant_found=0))

    # -- aggregation ------------------------------------------------------
    rows = [
        score_ranking(["a", "b"], {"a"}, 2, "document"),
        score_ranking(["c", "d"], {"a"}, 2, "document"),
        score_ranking(["a"], set(), 2, "document"),
    ]
    summary = aggregate(rows, 2, "document")
    check("aggregate averages the defined values", near(summary.recall, 0.5))
    check("aggregate ignores undefined values", summary.cases == 2,
          f"{summary.cases}")
    check("aggregate names the level and K",
          summary.level == "document" and summary.k == 2)
    check("aggregate over nothing is undefined",
          aggregate([], 2, "document").recall is None)
    check("aggregate ignores other cut-offs",
          aggregate(rows, 5, "document").cases == 0)
    check("a summary renders a row", "MRR=" in summary.row())
    check("summaries are immutable", MetricSummary.model_config["frozen"] is True)


# ===========================================================================
# 2. The dataset
# ===========================================================================
def test_dataset() -> None:
    print("\n2. Dataset integrity")

    check("the dataset covers groups A through H",
          sorted({case.group for case in CASES}) == list("ABCDEFGH"),
          str(sorted({case.group for case in CASES})))
    check("case ids are unique",
          len({case.case_id for case in CASES}) == len(CASES))
    check("every case describes itself", all(case.description for case in CASES))
    check("every case names its source",
          all(case.source in ("synthetic", "pcap") for case in CASES))
    check("the live subset is small",
          0 < len(cases_for_live()) <= 8, str(len(cases_for_live())))

    for case in CASES:
        unknown = (case.relevant_documents | case.irrelevant_documents) - CORPUS_DOCUMENT_IDS
        check(f"{case.case_id}: every label names a real document", not unknown,
              str(sorted(unknown)))
        overlap = case.relevant_documents & case.irrelevant_documents
        check(f"{case.case_id}: no document is both relevant and irrelevant",
              not overlap, str(sorted(overlap)))
        clash = case.expected_signals & case.forbidden_signals
        check(f"{case.case_id}: no signal is both expected and forbidden", not clash)
        check(f"{case.case_id}: relevant sections belong to relevant documents",
              all(document in case.relevant_documents
                  for document, _ in case.relevant_sections))

    check("the corpus label set matches the corpus on disk",
          _corpus_ids() == CORPUS_DOCUMENT_IDS,
          str(_corpus_ids() ^ CORPUS_DOCUMENT_IDS))

    # -- captures ---------------------------------------------------------
    for case in CASES:
        report = case.report()
        if report is None:
            check(f"{case.case_id}: absent capture is handled", case.source == "pcap")
            continue
        check(f"{case.case_id}: the capture has flows", len(report.flows) > 0)
        check(f"{case.case_id}: totals agree with the flows",
              report.totals.total_flows == len(report.flows)
              and report.totals.flows_included == len(report.flows))
        check(f"{case.case_id}: packet counts are derived, not invented",
              report.totals.total_packets
              == sum(f.packets_out + f.packets_in for f in report.flows))
        check(f"{case.case_id}: flow ids are unique",
              len({f.flow_id for f in report.flows}) == len(report.flows))

    first = CASES[0].report()
    second = CASES[0].report()
    check("captures are deterministic",
          first.model_dump_json() == second.model_dump_json())


def _corpus_ids() -> set[str]:
    from ai.rag.documents import load_corpus

    return {document.id for document in load_corpus()}


# ===========================================================================
# 3. Signal evaluation
# ===========================================================================
def test_signal_evaluation() -> None:
    print("\n3. Signal evaluation")

    result = harness.evaluate_signals()
    check("every case is reported", len(result["cases"]) == len(CASES))
    check("the scored count excludes skipped cases",
          result["cases_scored"] + result["cases_skipped"] == len(CASES))
    check("signal evaluation is deterministic",
          json.dumps(harness.evaluate_signals(), sort_keys=True, default=str)
          == json.dumps(result, sort_keys=True, default=str))

    for row in result["cases"]:
        if "skipped" in row:
            continue
        case = next(c for c in CASES if c.case_id == row["case"])
        found = set(row["found"])
        check(f"{row['case']}: missing is expected minus found",
              set(row["missing"]) == case.expected_signals - found)
        check(f"{row['case']}: false detections are forbidden and found",
              set(row["false_detections"]) == case.forbidden_signals & found)
        check(f"{row['case']}: additional signals are not counted as failures",
              set(row["additional_unlabelled"]).isdisjoint(case.forbidden_signals))

    check("the totals match the per-case rows",
          result["total_missing"] == sum(len(row["missing"]) for row in result["cases"]
                                         if "skipped" not in row))


# ===========================================================================
# 4. Sweep machinery, on a stub index
# ===========================================================================
class KeywordEncoder:
    """A deterministic stand-in, so the accounting can be checked with no model.

    Its vectors carry no meaning, and nothing in this file asserts that they
    do. It exists to prove the sweep plumbing adds up.
    """

    AXES = ("dns", "http", "scan", "browsing", "port", "upload")

    @property
    def name(self) -> str:
        return "test/keyword-6d"

    def dimension(self) -> int:
        return len(self.AXES)

    def encode(self, texts, normalize: bool):
        rows = []
        for text in texts:
            lowered = text.lower()
            row = [1.0 + float(lowered.count(axis)) for axis in self.AXES]
            if normalize:
                length = sum(v * v for v in row) ** 0.5
                row = [v / length for v in row]
            rows.append(row)
        return rows


def stub_index():
    """The real corpus, indexed with the stub encoder."""
    from ai.rag.chunking import chunk_corpus
    from ai.rag.documents import load_corpus

    embedder = EmbeddingModel(EmbeddingConfig(model_name="test/keyword-6d"),
                              encoder=KeywordEncoder())
    chunks = chunk_corpus(load_corpus())
    store = VectorStore("stub")
    store.add_many([VectorRecord(chunk=chunk, embedding=embedding)
                    for chunk, embedding in zip(chunks, embedder.embed_chunks(list(chunks)))])
    return store, embedder


def test_sweeps() -> None:
    print("\n4. Sweep machinery (stub index -- accounting only)")

    store, embedder = stub_index()

    retrieval = harness.evaluate_retrieval(store, embedder, k_values=(1, 3))
    check("retrieval evaluation covers every case",
          len(retrieval["cases"]) == len(CASES))
    check("it records the configuration it used",
          retrieval["config"]["final_top_k"] == RetrievalConfig().final_top_k)
    check("both cut-offs are summarised", len(retrieval["document_summary"]) == 2)
    check("document and section levels are distinguished",
          retrieval["document_summary"][0]["level"] == "document"
          and retrieval["section_summary"][0]["level"] == "chunk")
    check("retrieval evaluation is deterministic",
          json.dumps(harness.evaluate_retrieval(store, embedder, k_values=(1, 3)),
                     sort_keys=True, default=str)
          == json.dumps(retrieval, sort_keys=True, default=str))
    for row in retrieval["cases"]:
        if "skipped" in row:
            continue
        check(f"{row['case']}: missed documents are genuinely absent",
              set(row["relevant_missed"]).isdisjoint(set(row["documents_ranked"])))

    # -- budget accounting -------------------------------------------------
    budget = harness.evaluate_budget(store, embedder)
    check("the max_items sweep covers the grid",
          [row["max_items"] for row in budget["max_items"]]
          == list(harness.MAX_ITEMS_SWEEP))
    check("the max_chars sweep covers the grid",
          [row["max_chars"] for row in budget["max_chars"]]
          == list(harness.MAX_CHARS_SWEEP))
    check("the token sweep covers the grid",
          [row["max_total_tokens"] for row in budget["max_total_tokens"]]
          == list(harness.MAX_TOKENS_SWEEP))
    check("combined configurations are labelled",
          all(row["label"] and row["budget"] for row in budget["combined"]))

    for name, key in (("max_items", "max_items"), ("max_chars", "max_chars")):
        for row in budget[name]:
            for case_row in row["cases"]:
                check(f"{name}={row[key]} {case_row['case']}: retained and lost are disjoint",
                      not set(case_row["relevant_retained"]) & set(case_row["relevant_lost"]))
                check(f"{name}={row[key]} {case_row['case']}: retained is a subset of retrieved",
                      set(case_row["relevant_retained"])
                      <= set(case_row["relevant_retrieved"]))
                check(f"{name}={row[key]} {case_row['case']}: prompt exceeds the knowledge",
                      case_row["prompt_chars"] > case_row["knowledge_chars"])

    tighter = next(row for row in budget["max_items"] if row["max_items"] == 1)
    looser = next(row for row in budget["max_items"] if row["max_items"] == 6)
    check("a tighter item budget supplies no more than a looser one",
          tighter["supplied_total"] <= looser["supplied_total"])
    check("a tighter item budget excludes no fewer",
          tighter["excluded_total"] >= looser["excluded_total"])
    check("a tighter item budget retains no more relevant knowledge",
          tighter["relevant_retained_total"] <= looser["relevant_retained_total"])
    check("a tighter item budget produces a smaller mean prompt",
          tighter["mean_prompt_chars"] <= looser["mean_prompt_chars"])

    unbounded = next(row for row in budget["combined"] if row["label"] == "unbounded")
    check("the unbounded configuration excludes nothing",
          unbounded["excluded_total"] == 0)

    # -- thresholds --------------------------------------------------------
    thresholds = harness.evaluate_thresholds(store, embedder)
    check("the min_similarity sweep covers the grid",
          [row["min_similarity"] for row in thresholds["min_similarity"]]
          == list(harness.MIN_SIMILARITY_SWEEP))
    check("every shape parameter is swept",
          all(key in thresholds for key in
              ("per_query_top_k", "final_top_k", "max_per_document")))
    check("a higher threshold never retrieves more",
          all(a["retrieved_chunks"] >= b["retrieved_chunks"]
              for a, b in zip(thresholds["min_similarity"],
                              thresholds["min_similarity"][1:])),
          str([row["retrieved_chunks"] for row in thresholds["min_similarity"]]))
    check("threshold evaluation is deterministic",
          json.dumps(harness.evaluate_thresholds(store, embedder),
                     sort_keys=True, default=str)
          == json.dumps(thresholds, sort_keys=True, default=str))

    # -- recommendations ---------------------------------------------------
    rec = harness.recommend(retrieval, budget, thresholds)
    check("recommendations are produced from measurements", rec["available"] is True)
    check("every recommendation states its rule",
          all(rec[name]["why"] for name in
              ("per_query_top_k", "final_top_k", "max_per_document", "min_similarity",
               "max_items", "max_chars", "max_total_tokens", "max_flows")))
    check("recommendations are unavailable without measurements",
          harness.recommend(None, None, None)["available"] is False)
    check("recommending changes no default",
          KnowledgeContextConfig().max_items == 4
          and RetrievalConfig().min_similarity is None)


# ===========================================================================
# 5. Grounding checks and provider failure
# ===========================================================================
class _StubKnowledge:
    def __init__(self, refs: tuple[str, ...]) -> None:
        self._refs = refs
        self.items = tuple(range(len(refs)))

    def refs(self) -> tuple[str, ...]:
        return self._refs


class _StubOutcome:
    def __init__(self, analysis, refs=("K1", "K2"), ok=True, failure=None, detail=""):
        self.analysis = analysis
        self.knowledge = _StubKnowledge(refs)
        self.model = "test/model"
        self.ok = ok
        self.failure = failure
        self.detail = detail


def analysis(**overrides) -> AnalysisResult:
    data = dict(
        summary="Capture shows outbound HTTPS to well-known services.",
        observed_facts=["6 flows recorded."],
        interpretation=["Consistent with ordinary browsing [K1]."],
        uncertainties=["Payloads are encrypted."],
        traffic_type="web_browsing", risk_level="informational",
        risk_rationale="Ordinary traffic.", confidence=0.6,
        indicators=[], recommended_actions=[], notable_flow_ids=[],
        knowledge_refs=["K1"],
    )
    data.update(overrides)
    return AnalysisResult(**data)


def test_grounding() -> None:
    print("\n5. Grounding checks")

    case = next(c for c in CASES if c.case_id == "knowledge-conflicts-observation")
    report = case.report()

    clean = harness._grounding_row(case, _StubOutcome(analysis()), report)
    check("a well-grounded analysis has no problems", clean["problems"] == [],
          str(clean["problems"]))
    check("it records what was supplied and what was cited",
          clean["knowledge_supplied"] == ["K1", "K2"]
          and clean["knowledge_cited"] == ["K1"])
    check("it records flow-reference validity", clean["flow_refs_valid"] is True)
    check("it records knowledge-reference validity",
          clean["knowledge_refs_valid"] is True)

    invented = harness._grounding_row(
        case, _StubOutcome(analysis(knowledge_refs=["K9"]), refs=("K1",)), report)
    check("an invented citation is caught",
          any("not supplied" in problem for problem in invented["problems"]),
          str(invented["problems"]))

    inline = harness._grounding_row(
        case, _StubOutcome(analysis(interpretation=["Because [K7] says so."]),
                           refs=("K1",)), report)
    check("an inline citation that was never supplied is caught",
          any("inline citations" in problem for problem in inline["problems"]),
          str(inline["problems"]))

    fabricated = harness._grounding_row(
        case, _StubOutcome(analysis(notable_flow_ids=[999])), report)
    check("an invented flow id is caught", not fabricated["flow_refs_valid"])

    imported = harness._grounding_row(
        case, _StubOutcome(analysis(observed_facts=["1000 flows were observed."])),
        report)
    check("a number absent from the capture is caught",
          any("absent from the capture" in problem for problem in imported["problems"]),
          str(imported["problems"]))

    conflict = harness._grounding_row(
        case, _StubOutcome(analysis(observed_facts=["DNS queries were observed."])),
        report)
    check("knowledge inventing an observation is caught",
          any("absent from this capture" in problem for problem in conflict["problems"]),
          str(conflict["problems"]))

    silent = harness._grounding_row(
        case, _StubOutcome(analysis(uncertainties=[], interpretation=[])), report)
    check("empty uncertainties are caught",
          any("uncertainties is empty" in p for p in silent["problems"]))
    check("empty interpretation is caught",
          any("interpretation is empty" in p for p in silent["problems"]))

    escalated = harness._grounding_row(
        case, _StubOutcome(analysis(risk_level="high")), report)
    check("over-escalation beyond the case ceiling is caught",
          any("exceeds the case ceiling" in p for p in escalated["problems"]),
          str(escalated["problems"]))

    # -- provider failure is not a grounding failure -----------------------
    import ai.analyzer as analyzer_module

    from ai.llm_client import FailureReason

    class _Failed:
        ok = False
        failure = FailureReason.RATE_LIMITED
        detail = "APIStatusError; HTTP 413; message=Request too large; key gsk_SECRETVALUE123456"
        knowledge = _StubKnowledge(("K1",))
        analysis = None

    original = analyzer_module.analyze_capture
    analyzer_module.analyze_capture = lambda *a, **k: _Failed()
    try:
        store, embedder = stub_index()
        live = harness.evaluate_live(store, embedder)
    finally:
        analyzer_module.analyze_capture = original

    check("a provider failure is recorded as a provider failure",
          live["provider_failures"] == live["attempted"] > 0,
          f"{live['provider_failures']}/{live['attempted']}")
    check("a provider failure produces no grounding result", live["analysed"] == 0)
    check("a provider failure is not counted as clean", live["grounding_clean"] == 0)
    check("the failure detail is kept", all("rate_limited" == row["provider_failure"]
                                            for row in live["cases"]))
    check("the failure detail is redacted",
          all("gsk_SECRETVALUE" not in row["detail"] for row in live["cases"]),
          str(live["cases"][0]["detail"]))


# ===========================================================================
# 6. Report, JSON and privacy
# ===========================================================================
def test_output() -> None:
    print("\n6. Report output")

    result = harness.run(live=False)
    check("run() returns every section",
          set(result) >= {"dataset", "signals", "retrieval", "budget", "thresholds",
                          "live", "index", "recommendations"})
    check("the live section is skipped unless requested",
          "skipped" in result["live"], str(result["live"])[:80])
    check("the dataset section names every case",
          len(result["dataset"]["case_detail"]) == len(CASES))
    check("label errors are reported as a list",
          result["dataset"]["label_errors"] == [])

    rendered = harness.render(result)
    for heading in ("1. DATASET", "2. SIGNAL EXTRACTION", "7. LIVE LLM GROUNDING",
                    "8. FAILURE CASES", "9. RECOMMENDED DEFAULTS", "10. LIMITATIONS"):
        check(f"the report contains {heading!r}", heading in rendered)
    check("the report states its limitations",
          len(harness.LIMITATIONS) >= 5 and harness.LIMITATIONS[0] in rendered)
    check("the report is deterministic", harness.render(result) == rendered)
    check("the report contains no timestamp",
          not any(word in rendered.lower()
                  for word in ("generated at", "timestamp", "elapsed")))

    # -- JSON --------------------------------------------------------------
    encoded = json.dumps(result, indent=2, sort_keys=True, default=str)
    check("the result serialises to JSON", json.loads(encoded)["dataset"]["cases"] == 8)
    check("JSON serialisation is deterministic",
          json.dumps(harness.run(live=False), indent=2, sort_keys=True, default=str)
          == encoded)
    check("JSON round-trips", json.loads(encoded)["signals"]["cases_scored"]
          == result["signals"]["cases_scored"])

    # -- privacy -----------------------------------------------------------
    os.environ["DPI_EVAL_TEST_SECRET"] = "sk-must-never-appear"
    try:
        fresh = harness.render(harness.run(live=False))
        check("no environment value leaks into the report",
              "sk-must-never-appear" not in fresh)
    finally:
        os.environ.pop("DPI_EVAL_TEST_SECRET", None)

    for marker in ("sk-", "gsk_", "api_key", "API_KEY", "Bearer "):
        check(f"no {marker!r} appears in the report", marker not in rendered)
    check("no address appears in the report", not IPV4.search(rendered),
          str(IPV4.findall(rendered)[:3]))
    check("the redactor removes a key-shaped token",
          "gsk_SECRETVALUE" not in harness._redact("key gsk_SECRETVALUE1234567890"))

    # -- CLI ---------------------------------------------------------------
    from contextlib import redirect_stdout
    import io as _io

    plain_quiet = _io.StringIO()
    with redirect_stdout(plain_quiet):
        plain_exit = harness.main([])

    json_quiet = _io.StringIO()
    with redirect_stdout(json_quiet):
        json_exit = harness.main(["--json"])

    printed = plain_quiet.getvalue() + json_quiet.getvalue()
    json_output = json_quiet.getvalue().strip()

    check("the harness exits 0 on a normal run", plain_exit == 0)
    check("the harness exits 0 in JSON mode", json_exit == 0)

    try:
        parsed_json = json.loads(json_output)
        json_valid = isinstance(parsed_json, dict)
    except json.JSONDecodeError:
        json_valid = False

    check("JSON mode emits parseable JSON", json_valid)

    check("the CLI output carries no secret", "sk-" not in printed
          and "gsk_" not in printed)
    
    # -- isolation ---------------------------------------------------------
    source = Path("run_rag_evaluation.py").read_text(encoding="utf-8")
    check("the harness never writes to the corpus",
          "write_text" not in source and "open(" not in source)
    for package in ("dpi/dpi_engine.py", "ai/analyzer.py", "ai/rag/retrieval.py"):
        text = Path(package).read_text(encoding="utf-8")
        check(f"{package} does not import the evaluation harness",
              "evaluation" not in text.replace("evaluation step", "").replace(
                  "evaluation)", "").replace("evaluation.", ""))


# ===========================================================================
def main() -> int:
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print("RAG step 8 -- evaluation harness tests")

    test_metrics()
    test_dataset()
    test_signal_evaluation()
    test_sweeps()
    test_grounding()
    test_output()

    total = _passed + _failed
    print(f"\n{_passed}/{total} checks passed")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
