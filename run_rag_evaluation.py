#!/usr/bin/env python3
"""Evaluate the RAG + AI analysis pipeline against a fixed dataset.

    python run_rag_evaluation.py            # console report
    python run_rag_evaluation.py --json     # machine-readable
    python run_rag_evaluation.py --live     # add a small live-LLM grounding pass

What this is for
----------------
Telling us where the pipeline is wrong, not making it look right. Every number
here is measured from :mod:`evaluation.cases`, whose labels were written by
hand from the corpus and the DPI schema before any retrieval was run. Nothing
in this file adjusts a label, a default or a threshold; it reports, and the
recommendations at the end are derived from the measurements by stated rules
that a reader can check.

Determinism
-----------
Everything except the live-LLM section is deterministic: same corpus, same
model, same numbers. No clock is read, no sampling is done, and the live pass
is **opt-in** so a routine run neither varies nor spends quota.

What needs what
---------------
* Signal evaluation and the metric arithmetic need nothing at all.
* Retrieval, budget and threshold sweeps need ``sentence-transformers`` and the
  ``BAAI/bge-small-en-v1.5`` weights. Without them those sections state that
  they were skipped and why -- they are never estimated or faked.
* The live pass additionally needs ``GROQ_API_KEY`` and ``--live``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Sequence

# Import the package so its console-encoding fix is applied before any output.
import dpi  # noqa: F401
from ai.schemas import CaptureReport
from evaluation.cases import CASES, CORPUS_DOCUMENT_IDS, EvaluationCase, cases_for_live
from evaluation.metrics import (
    DEFAULT_K_VALUES,
    MetricSummary,
    RetrievalMetrics,
    aggregate,
    score_ranking,
)

# ---------------------------------------------------------------------------
# Sweep grids.  One variable at a time from the shipped defaults, plus a few
# combinations -- a full Cartesian product would be hundreds of runs and would
# not answer a question the one-at-a-time sweeps leave open.
# ---------------------------------------------------------------------------
MAX_ITEMS_SWEEP: tuple[int, ...] = (1, 2, 4, 6)
MAX_CHARS_SWEEP: tuple[int, ...] = (1000, 2000, 3000, 4000, 6000)
MAX_TOKENS_SWEEP: tuple[int | None, ...] = (300, 600, 900, 1200, None)
MIN_SIMILARITY_SWEEP: tuple[float | None, ...] = (
    None, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
)
PER_QUERY_TOP_K_SWEEP: tuple[int, ...] = (2, 4, 6)
FINAL_TOP_K_SWEEP: tuple[int, ...] = (4, 8, 12)
MAX_PER_DOCUMENT_SWEEP: tuple[int | None, ...] = (1, 2, 3, None)

#: Cut-off used when a single headline number is wanted.
HEADLINE_K = 8

_SECRET_PATTERN = re.compile(r"(?i)\b(?:sk|gsk)[-_][A-Za-z0-9_\-]{6,}")
_NUMBER_PATTERN = re.compile(r"\b\d{2,}\b")


# ===========================================================================
# Small helpers
# ===========================================================================
def _section(title: str) -> str:
    return f"\n{'=' * 74}\n{title}\n{'=' * 74}"


def _show(value: float | None, places: int = 3) -> str:
    return " n/a " if value is None else f"{value:.{places}f}"


def _document_ranking(chunks: Sequence[Any]) -> list[str]:
    return [chunk.document_id for chunk in chunks]


def _section_ranking(chunks: Sequence[Any]) -> list[tuple[str, str]]:
    return [(chunk.document_id, chunk.section) for chunk in chunks]


# ===========================================================================
# Phase 4 -- signal extraction
# ===========================================================================
def evaluate_signals() -> dict[str, Any]:
    """Compare the deterministic signal extractor against the labels.

    Three outcomes are tracked separately, because they mean different things:
    a **missing** expected signal is a detection failure, a **forbidden**
    signal is a false detection, and an **additional** signal is merely
    unlabelled -- the label set is a floor and a prohibition, not an
    exhaustive prediction.
    """
    from ai.rag.signals import extract_signals

    rows: list[dict[str, Any]] = []
    for case in CASES:
        report = case.report()
        if report is None:
            rows.append({"case": case.case_id, "group": case.group, "skipped":
                         "capture unavailable"})
            continue

        signal_report = extract_signals(report)
        found = set(signal_report.types())
        severities = {s.signal_type.value: s.severity.value
                      for s in signal_report.signals}

        rows.append({
            "case": case.case_id,
            "group": case.group,
            "source": case.source,
            "flows": len(report.flows),
            "signal_count": signal_report.signal_count,
            "found": sorted(found),
            "expected_found": sorted(case.expected_signals & found),
            "missing": sorted(case.expected_signals - found),
            "false_detections": sorted(case.forbidden_signals & found),
            "additional_unlabelled": sorted(found - case.expected_signals),
            "severities": dict(sorted(severities.items())),
            "baseline_is_informational":
                severities.get("baseline_web_browsing") == "info",
        })

    scored = [row for row in rows if "skipped" not in row]
    return {
        "cases": rows,
        "cases_scored": len(scored),
        "cases_skipped": len(rows) - len(scored),
        "cases_fully_correct": sum(1 for row in scored
                                   if not row["missing"] and not row["false_detections"]),
        "total_missing": sum(len(row["missing"]) for row in scored),
        "total_false_detections": sum(len(row["false_detections"]) for row in scored),
    }


# ===========================================================================
# Index construction (everything below needs it)
# ===========================================================================
class IndexUnavailable(Exception):
    """The embedding model or corpus could not be prepared."""


def build_index() -> tuple[Any, Any, int]:
    """Load the corpus, embed it once, and return ``(store, embedder, chunks)``."""
    try:
        from ai.rag.chunking import chunk_corpus
        from ai.rag.documents import load_corpus
        from ai.rag.embeddings import (
            EmbeddingConfig,
            EmbeddingModel,
            ModelUnavailableError,
            sentence_transformers_available,
        )
        from ai.rag.vector_store import VectorRecord, VectorStore
    except ImportError as exc:
        raise IndexUnavailable(
            f"optional RAG dependencies are not installed ({exc}); "
            "pip install -r requirements-rag.txt"
        ) from exc

    if not sentence_transformers_available():
        raise IndexUnavailable("sentence-transformers is not installed")

    embedder = EmbeddingModel(EmbeddingConfig())
    try:
        embedder.load()
    except ModelUnavailableError as exc:
        raise IndexUnavailable(f"the embedding model could not be loaded: {exc}") from exc

    corpus = load_corpus()
    chunks = chunk_corpus(corpus)
    store = VectorStore("evaluation")
    store.add_many([VectorRecord(chunk=chunk, embedding=embedding)
                    for chunk, embedding in zip(chunks, embedder.embed_chunks(list(chunks)))])
    return store, embedder, len(chunks)


def _retrieve(case: EvaluationCase, report: CaptureReport, store: Any, embedder: Any,
              retrieval_config: Any) -> Any:
    from ai.rag.retrieval import retrieve_for_signals
    from ai.rag.signals import extract_signals

    return retrieve_for_signals(extract_signals(report), store, embedder, retrieval_config)


# ===========================================================================
# Phase 3 -- retrieval metrics
# ===========================================================================
def evaluate_retrieval(store: Any, embedder: Any,
                       k_values: Sequence[int] = DEFAULT_K_VALUES) -> dict[str, Any]:
    """Score retrieval at the shipped defaults, per case and in aggregate."""
    from ai.rag.retrieval import RetrievalConfig

    config = RetrievalConfig()
    per_case: list[dict[str, Any]] = []
    document_rows: list[RetrievalMetrics] = []
    section_rows: list[RetrievalMetrics] = []

    for case in CASES:
        report = case.report()
        if report is None:
            per_case.append({"case": case.case_id, "skipped": "capture unavailable"})
            continue

        retrieval = _retrieve(case, report, store, embedder, config)
        documents = _document_ranking(retrieval.chunks)
        sections = _section_ranking(retrieval.chunks)

        case_documents = []
        case_sections = []
        for k in k_values:
            doc_metric = score_ranking(documents, case.relevant_documents, k,
                                       "document", case.irrelevant_documents)
            sec_metric = score_ranking(sections, case.relevant_sections, k,
                                       "chunk", dedupe=False)
            document_rows.append(doc_metric)
            section_rows.append(sec_metric)
            case_documents.append(doc_metric)
            case_sections.append(sec_metric)

        per_case.append({
            "case": case.case_id,
            "group": case.group,
            "retrieved_chunks": retrieval.chunk_count,
            "queries": retrieval.query_count,
            "documents_ranked": list(dict.fromkeys(documents)),
            "top_similarity": round(retrieval.chunks[0].similarity, 4)
            if retrieval.chunks else None,
            "min_similarity": round(retrieval.chunks[-1].similarity, 4)
            if retrieval.chunks else None,
            "document_metrics": [m.model_dump() for m in case_documents],
            "section_metrics": [m.model_dump() for m in case_sections],
            "relevant_missed": sorted(case.relevant_documents - set(documents)),
            "irrelevant_retrieved": sorted(case.irrelevant_documents & set(documents)),
        })

    return {
        "config": {"per_query_top_k": config.per_query_top_k,
                   "final_top_k": config.final_top_k,
                   "min_similarity": config.min_similarity,
                   "max_per_document": config.max_per_document},
        "k_values": list(k_values),
        "cases": per_case,
        "document_summary": [aggregate(document_rows, k, "document").model_dump()
                             for k in k_values],
        "section_summary": [aggregate(section_rows, k, "chunk").model_dump()
                            for k in k_values],
    }


# ===========================================================================
# Phase 5 -- context budget
# ===========================================================================
def _budget_row(case: EvaluationCase, retrieval: Any, context: Any,
                report: CaptureReport) -> dict[str, Any]:
    """Measure one budgeted context against the case's relevance labels."""
    from ai.prompts import build_messages

    supplied = {item.document_id for item in context.items}
    excluded = {dropped.document_id for dropped in context.excluded}
    retrieved = {chunk.document_id for chunk in retrieval.chunks}

    prompt = build_messages(report, None, context.text or None)
    prompt_chars = sum(len(message["content"]) for message in prompt)

    return {
        "case": case.case_id,
        "supplied_items": len(context.items),
        "excluded_items": context.dropped_items,
        "estimated_tokens": context.estimated_tokens,
        "knowledge_chars": len(context.text),
        "prompt_chars": prompt_chars,
        "relevant_retrieved": sorted(case.relevant_documents & retrieved),
        "relevant_retained": sorted(case.relevant_documents & supplied),
        "relevant_lost": sorted((case.relevant_documents & excluded) - supplied),
        "irrelevant_supplied": sorted(case.irrelevant_documents & supplied),
    }


def _sweep(store: Any, embedder: Any, retrieval_config: Any, context_config: Any,
           k: int = HEADLINE_K) -> dict[str, Any]:
    """Run every case at one configuration and summarise it."""
    from ai.rag.context import build_knowledge_context

    rows: list[dict[str, Any]] = []
    document_rows: list[RetrievalMetrics] = []
    for case in CASES:
        report = case.report()
        if report is None:
            continue
        retrieval = _retrieve(case, report, store, embedder, retrieval_config)
        context = build_knowledge_context(retrieval, context_config)
        rows.append(_budget_row(case, retrieval, context, report))
        supplied_documents = [item.document_id for item in context.items]
        document_rows.append(score_ranking(supplied_documents, case.relevant_documents,
                                           k, "document", case.irrelevant_documents))

    summary = aggregate(document_rows, k, "document")
    return {
        "cases": rows,
        "supplied_total": sum(row["supplied_items"] for row in rows),
        "excluded_total": sum(row["excluded_items"] for row in rows),
        "relevant_retained_total": sum(len(row["relevant_retained"]) for row in rows),
        "relevant_lost_total": sum(len(row["relevant_lost"]) for row in rows),
        "mean_estimated_tokens": round(
            sum(row["estimated_tokens"] for row in rows) / len(rows), 1) if rows else 0.0,
        "mean_prompt_chars": round(
            sum(row["prompt_chars"] for row in rows) / len(rows), 1) if rows else 0.0,
        "supplied_recall": summary.recall,
        "supplied_precision": summary.precision,
        "irrelevant_supplied_total": sum(len(row["irrelevant_supplied"]) for row in rows),
    }


def evaluate_budget(store: Any, embedder: Any) -> dict[str, Any]:
    """One-variable-at-a-time sweeps over the context budget, plus combinations."""
    from ai.rag.context import KnowledgeContextConfig
    from ai.rag.retrieval import RetrievalConfig

    retrieval_config = RetrievalConfig()
    baseline = KnowledgeContextConfig()

    items: list[dict[str, Any]] = []
    for max_items in MAX_ITEMS_SWEEP:
        config = KnowledgeContextConfig(max_items=max_items,
                                        max_chars=10 ** 6, max_total_tokens=None)
        items.append({"max_items": max_items,
                      **_sweep(store, embedder, retrieval_config, config)})

    chars: list[dict[str, Any]] = []
    for max_chars in MAX_CHARS_SWEEP:
        config = KnowledgeContextConfig(max_items=99, max_chars=max_chars,
                                        max_total_tokens=None)
        chars.append({"max_chars": max_chars,
                      **_sweep(store, embedder, retrieval_config, config)})

    tokens: list[dict[str, Any]] = []
    for max_tokens in MAX_TOKENS_SWEEP:
        config = KnowledgeContextConfig(max_items=99, max_chars=10 ** 6,
                                        max_total_tokens=max_tokens)
        tokens.append({"max_total_tokens": max_tokens,
                       **_sweep(store, embedder, retrieval_config, config)})

    combined: list[dict[str, Any]] = []
    for label, config in (
        ("shipped default", baseline),
        ("tight (2 / 1500 / 400)",
         KnowledgeContextConfig(max_items=2, max_chars=1500, max_total_tokens=400)),
        ("generous (6 / 6000 / 1800)",
         KnowledgeContextConfig(max_items=6, max_chars=6000, max_total_tokens=1800)),
        ("unbounded", KnowledgeContextConfig(max_items=99, max_chars=10 ** 6,
                                             max_total_tokens=None)),
    ):
        combined.append({"label": label, "budget": config.describe(),
                         **_sweep(store, embedder, retrieval_config, config)})

    return {"max_items": items, "max_chars": chars, "max_total_tokens": tokens,
            "combined": combined}


# ===========================================================================
# Phase 6 -- min_similarity, and the other retrieval parameters
# ===========================================================================
def evaluate_thresholds(store: Any, embedder: Any,
                        k: int = HEADLINE_K) -> dict[str, Any]:
    """Sweep ``min_similarity`` and the three retrieval shape parameters."""
    from ai.rag.context import KnowledgeContextConfig
    from ai.rag.retrieval import RetrievalConfig

    context_config = KnowledgeContextConfig()

    def measure(config: RetrievalConfig) -> dict[str, Any]:
        from ai.rag.context import build_knowledge_context

        document_rows: list[RetrievalMetrics] = []
        retrieved = 0
        irrelevant = 0
        retained = 0
        empty_cases = 0
        for case in CASES:
            report = case.report()
            if report is None:
                continue
            retrieval = _retrieve(case, report, store, embedder, config)
            documents = _document_ranking(retrieval.chunks)
            document_rows.append(score_ranking(documents, case.relevant_documents, k,
                                               "document", case.irrelevant_documents))
            retrieved += retrieval.chunk_count
            irrelevant += len(case.irrelevant_documents & set(documents))
            if retrieval.chunk_count == 0:
                empty_cases += 1
            context = build_knowledge_context(retrieval, context_config)
            retained += len(case.relevant_documents
                            & {item.document_id for item in context.items})

        summary = aggregate(document_rows, k, "document")
        return {
            "retrieved_chunks": retrieved,
            "irrelevant_documents": irrelevant,
            "empty_cases": empty_cases,
            "recall": summary.recall,
            "precision": summary.precision,
            "hit": summary.hit,
            "mrr": summary.mrr,
            "relevant_retained_after_budget": retained,
        }

    thresholds = [{"min_similarity": value,
                   **measure(RetrievalConfig(min_similarity=value))}
                  for value in MIN_SIMILARITY_SWEEP]
    per_query = [{"per_query_top_k": value,
                  **measure(RetrievalConfig(per_query_top_k=value))}
                 for value in PER_QUERY_TOP_K_SWEEP]
    final_top = [{"final_top_k": value, **measure(RetrievalConfig(final_top_k=value))}
                 for value in FINAL_TOP_K_SWEEP]
    per_document = [{"max_per_document": value,
                     **measure(RetrievalConfig(max_per_document=value))}
                    for value in MAX_PER_DOCUMENT_SWEEP]

    return {"min_similarity": thresholds, "per_query_top_k": per_query,
            "final_top_k": final_top, "max_per_document": per_document}


# ===========================================================================
# Phase 7 -- live grounding
# ===========================================================================
def evaluate_live(store: Any, embedder: Any) -> dict[str, Any]:
    """Send a small representative subset to the configured provider.

    Provider failure is recorded as provider failure. It is never counted as a
    retrieval or grounding result, because it is evidence about a quota, not
    about this code.
    """
    from ai.analyzer import analyze_capture
    from ai.config import AIConfig
    from ai.rag.pipeline import KnowledgePipeline

    pipeline = KnowledgePipeline.from_index(store, embedder)
    config = AIConfig.from_env(provider="groq")
    rows: list[dict[str, Any]] = []

    for case in cases_for_live():
        report = case.report()
        if report is None:
            rows.append({"case": case.case_id, "skipped": "capture unavailable"})
            continue

        import ai.analyzer as analyzer_module

        original = analyzer_module.build_capture_report
        analyzer_module.build_capture_report = lambda *a, **k: report
        try:
            outcome = analyze_capture(object(), report.capture_name, config,
                                      rag=pipeline)
        finally:
            analyzer_module.build_capture_report = original

        if not outcome.ok:
            rows.append({
                "case": case.case_id,
                "provider_failure": outcome.failure.value if outcome.failure else "?",
                "detail": _redact(outcome.detail)[:200],
                "knowledge_supplied": len(outcome.knowledge.items)
                if outcome.knowledge else 0,
            })
            continue

        rows.append(_grounding_row(case, outcome, report))

    analysed = [row for row in rows if "provider_failure" not in row
                and "skipped" not in row]
    return {
        "cases": rows,
        "attempted": len(rows),
        "analysed": len(analysed),
        "provider_failures": sum(1 for row in rows if "provider_failure" in row),
        "grounding_clean": sum(1 for row in analysed if not row["problems"]),
    }


def _grounding_row(case: EvaluationCase, outcome: Any,
                   report: CaptureReport) -> dict[str, Any]:
    """The seven grounding checks, for one successful analysis."""
    analysis = outcome.analysis
    supplied = outcome.knowledge.refs() if outcome.knowledge else ()
    capture_json = report.model_dump_json()
    problems: list[str] = []

    flow_problems = analysis.validate_flow_references(report.flow_ids())
    if flow_problems:
        problems.extend(flow_problems)

    citation_problems = analysis.validate_knowledge_references(supplied)
    if citation_problems:
        problems.extend(citation_problems)

    # Numbers stated as observed must appear in the capture the model was
    # given.  Crude -- a number can coincide -- but it catches the failure that
    # matters: a figure imported from reference knowledge.
    unsupported: list[str] = []
    for fact in analysis.observed_facts:
        for number in _NUMBER_PATTERN.findall(fact):
            if number not in capture_json:
                unsupported.append(f"{number} (in: {fact[:60]})")
    if unsupported:
        problems.append(f"observed_facts cite numbers absent from the capture: "
                        f"{unsupported[:3]}")

    inline = set(re.findall(r"\[(K[1-9][0-9]*)\]", " ".join(
        analysis.interpretation + [analysis.risk_rationale]
        + [action.description for action in analysis.recommended_actions])))
    invented_inline = sorted(inline - set(supplied))
    if invented_inline:
        problems.append(f"inline citations not supplied: {invented_inline}")

    forbidden_hits = [term for term in case.forbidden_fact_terms
                      if any(term in fact.lower() for fact in analysis.observed_facts)]
    if forbidden_hits:
        problems.append(f"observed_facts mention {forbidden_hits}, absent from this capture")

    if not analysis.uncertainties:
        problems.append("uncertainties is empty")
    if not analysis.interpretation:
        problems.append("interpretation is empty")

    order = ["informational", "low", "medium", "high", "unknown"]
    over_escalated = (case.max_risk is not None
                      and analysis.risk_level.value in order
                      and case.max_risk in order
                      and order.index(analysis.risk_level.value) > order.index(case.max_risk)
                      and analysis.risk_level.value != "unknown")
    if over_escalated:
        problems.append(f"risk {analysis.risk_level.value} exceeds the case ceiling "
                        f"{case.max_risk}")

    return {
        "case": case.case_id,
        "model": outcome.model,
        "risk_level": analysis.risk_level.value,
        "traffic_type": analysis.traffic_type.value,
        "confidence": analysis.confidence,
        "knowledge_supplied": list(supplied),
        "knowledge_cited": list(analysis.knowledge_refs),
        "observed_facts": len(analysis.observed_facts),
        "interpretation": len(analysis.interpretation),
        "uncertainties": len(analysis.uncertainties),
        "flow_refs_valid": not flow_problems,
        "knowledge_refs_valid": not citation_problems,
        "problems": problems,
    }


def _redact(text: str) -> str:
    return _SECRET_PATTERN.sub("[redacted]", text or "")


# ===========================================================================
# Phase 8 -- recommendations derived from the measurements
# ===========================================================================
def recommend(retrieval: dict[str, Any] | None, budget: dict[str, Any] | None,
              thresholds: dict[str, Any] | None) -> dict[str, Any]:
    """Derive default recommendations from what was measured.

    Every rule is stated beside its answer, so a reader can disagree with the
    rule rather than having to trust the number. Nothing here changes a
    default; recommending and applying are separate acts, and only the second
    one needs a human.
    """
    if budget is None or thresholds is None or retrieval is None:
        return {"available": False,
                "reason": "the model-dependent sweeps did not run"}

    def best_min_similarity() -> tuple[Any, str]:
        rows = thresholds["min_similarity"]
        recalls = [row["recall"] for row in rows if row["recall"] is not None]
        if not recalls:
            return None, "no recall was measurable"
        best = max(recalls)
        # Largest threshold that keeps full recall and empties no case.
        qualifying = [row for row in rows
                      if row["recall"] is not None and row["recall"] >= best - 1e-9
                      and row["empty_cases"] == 0]
        if not qualifying:
            return None, "every threshold lost recall; keep None"
        chosen = max(qualifying,
                     key=lambda row: (-1.0 if row["min_similarity"] is None
                                      else row["min_similarity"]))
        return (chosen["min_similarity"],
                f"largest threshold with no recall loss (recall {best:.3f}) "
                f"and no empty case")

    def smallest_keeping(rows: list[dict[str, Any]], key: str) -> tuple[Any, str]:
        retained = [row["relevant_retained_total"] for row in rows]
        if not retained:
            return None, "no rows"
        best = max(retained)
        keeping = [row for row in rows if row["relevant_retained_total"] == best]
        chosen = min(keeping, key=lambda row: (10 ** 9 if row[key] is None else row[key]))
        return (chosen[key],
                f"smallest value retaining all {best} relevant document(s) "
                f"the sweep ever retained")

    def best_shape(rows: list[dict[str, Any]], key: str) -> tuple[Any, str]:
        scored = [row for row in rows if row["recall"] is not None]
        if not scored:
            return None, "no recall was measurable"
        best = max(row["recall"] for row in scored)
        keeping = [row for row in scored if row["recall"] >= best - 1e-9]
        chosen = min(keeping, key=lambda row: (10 ** 9 if row[key] is None else row[key]))
        return chosen[key], f"smallest value reaching the best recall ({best:.3f})"

    items_value, items_why = smallest_keeping(budget["max_items"], "max_items")
    chars_value, chars_why = smallest_keeping(budget["max_chars"], "max_chars")
    tokens_value, tokens_why = smallest_keeping(budget["max_total_tokens"],
                                                "max_total_tokens")
    similarity_value, similarity_why = best_min_similarity()
    per_query_value, per_query_why = best_shape(thresholds["per_query_top_k"],
                                                "per_query_top_k")
    final_value, final_why = best_shape(thresholds["final_top_k"], "final_top_k")
    document_value, document_why = best_shape(thresholds["max_per_document"],
                                              "max_per_document")

    largest_prompt = max((row["prompt_chars"]
                          for entry in budget["combined"] for row in entry["cases"]),
                         default=0)

    return {
        "available": True,
        "per_query_top_k": {"value": per_query_value, "why": per_query_why},
        "final_top_k": {"value": final_value, "why": final_why},
        "max_per_document": {"value": document_value, "why": document_why},
        "min_similarity": {"value": similarity_value, "why": similarity_why},
        "max_items": {"value": items_value, "why": items_why},
        "max_chars": {"value": chars_value, "why": chars_why},
        "max_total_tokens": {"value": tokens_value, "why": tokens_why},
        "max_flows": {
            "value": None,
            "why": (f"largest prompt measured was {largest_prompt} characters "
                    f"(~{largest_prompt // 4} tokens); no capture-size problem was "
                    "observed in this dataset, so no change is recommended"),
        },
    }


# ===========================================================================
# Report assembly
# ===========================================================================
def run(live: bool = False) -> dict[str, Any]:
    """Run every phase that can run, and return the whole result."""
    result: dict[str, Any] = {
        "dataset": {
            "cases": len(CASES),
            "groups": sorted({case.group for case in CASES}),
            "sources": {source: sum(1 for case in CASES if case.source == source)
                        for source in sorted({case.source for case in CASES})},
            "corpus_documents": sorted(CORPUS_DOCUMENT_IDS),
            "live_subset": [case.case_id for case in cases_for_live()],
            "label_errors": sorted(
                doc for case in CASES
                for doc in (case.relevant_documents | case.irrelevant_documents)
                if doc not in CORPUS_DOCUMENT_IDS),
            "case_detail": [{
                "case": case.case_id, "group": case.group, "source": case.source,
                "description": case.description,
                "expected_signals": sorted(case.expected_signals),
                "forbidden_signals": sorted(case.forbidden_signals),
                "relevant_documents": sorted(case.relevant_documents),
                "relevant_sections": sorted(map(list, case.relevant_sections)),
                "irrelevant_documents": sorted(case.irrelevant_documents),
                "live": case.live, "max_risk": case.max_risk,
            } for case in CASES],
        },
        "signals": evaluate_signals(),
        "retrieval": None,
        "budget": None,
        "thresholds": None,
        "live": None,
        "index": {"available": False, "reason": ""},
    }

    try:
        store, embedder, chunk_count = build_index()
    except IndexUnavailable as exc:
        result["index"] = {"available": False, "reason": str(exc)}
        result["recommendations"] = recommend(None, None, None)
        # The live pass needs the same index, so it cannot run either.  Saying
        # so is better than leaving the section null and letting a reader guess.
        result["live"] = {"skipped": f"the knowledge index is unavailable: {exc}"}
        return result

    result["index"] = {"available": True, "chunks": chunk_count,
                       "dimension": store.dimension, "model": store.model_name}
    result["retrieval"] = evaluate_retrieval(store, embedder)
    result["budget"] = evaluate_budget(store, embedder)
    result["thresholds"] = evaluate_thresholds(store, embedder)
    result["recommendations"] = recommend(result["retrieval"], result["budget"],
                                          result["thresholds"])

    if live:
        # Ask AIConfig, not os.environ: the key normally arrives through .env,
        # which is loaded by from_env() and is invisible to a bare getenv.
        # Gate on the same provider the live pass will actually use, so the
        # check and the run can never disagree about which key matters.
        from ai.config import AIConfig

        config = AIConfig.from_env(provider="groq")

        if not config.has_api_key():
            result["live"] = {
                "skipped": (f"no API key for {config.spec.label}; set "
                            f"{config.spec.api_key_env} in the environment or .env"),
            }
        else:
            result["live"] = evaluate_live(store, embedder)
    else:
        result["live"] = {"skipped": "not requested; pass --live to run it"}

    return result


def render(result: dict[str, Any]) -> str:
    """The console report."""
    out: list[str] = []
    data = result["dataset"]

    out.append(_section("1. DATASET"))
    out.append(f"  cases:      {data['cases']} (groups {', '.join(data['groups'])})")
    out.append(f"  sources:    " + ", ".join(f"{n} {s}" for s, n in data["sources"].items()))
    out.append(f"  corpus:     {len(data['corpus_documents'])} documents")
    out.append(f"  live subset:{len(data['live_subset'])} case(s)")
    if data["label_errors"]:
        out.append(f"  LABEL ERRORS: {data['label_errors']}")
    for case in data["case_detail"]:
        out.append(f"    {case['group']} {case['case']:<34} {case['source']:<10} "
                   f"{len(case['relevant_documents'])} relevant, "
                   f"{len(case['irrelevant_documents'])} irrelevant")

    signals = result["signals"]
    out.append(_section("2. SIGNAL EXTRACTION"))
    out.append(f"  {signals['cases_fully_correct']}/{signals['cases_scored']} cases "
               f"fully correct   missing={signals['total_missing']}   "
               f"false detections={signals['total_false_detections']}")
    for row in signals["cases"]:
        if "skipped" in row:
            out.append(f"    {row['case']:<34} skipped: {row['skipped']}")
            continue
        flag = "ok " if not row["missing"] and not row["false_detections"] else "!! "
        out.append(f"  {flag}{row['case']:<34} {row['signal_count']} signal(s): "
                   f"{', '.join(row['found'])}")
        if row["missing"]:
            out.append(f"      MISSING: {row['missing']}")
        if row["false_detections"]:
            out.append(f"      FALSE:   {row['false_detections']}")
        if row["additional_unlabelled"]:
            out.append(f"      also (unlabelled): {row['additional_unlabelled']}")

    index = result["index"]
    if not index["available"]:
        out.append(_section("3-6. RETRIEVAL, BUDGET AND THRESHOLD SWEEPS"))
        out.append(f"  SKIPPED: {index['reason']}")
        out.append("  These sections need the real embedding model; they are never")
        out.append("  estimated. Install requirements-rag.txt and re-run.")
    else:
        out.extend(_render_retrieval(result["retrieval"]))
        out.extend(_render_budget(result["budget"]))
        out.extend(_render_thresholds(result["thresholds"]))

    out.append(_section("7. LIVE LLM GROUNDING"))
    live = result["live"]
    if "skipped" in live:
        out.append(f"  SKIPPED: {live['skipped']}")
    else:
        out.append(f"  {live['analysed']}/{live['attempted']} analysed, "
                   f"{live['provider_failures']} provider failure(s), "
                   f"{live['grounding_clean']} clean")
        for row in live["cases"]:
            if "provider_failure" in row:
                out.append(f"    {row['case']:<34} PROVIDER FAILURE "
                           f"[{row['provider_failure']}] {row['detail'][:90]}")
            elif "skipped" in row:
                out.append(f"    {row['case']:<34} skipped: {row['skipped']}")
            else:
                out.append(f"    {row['case']:<34} risk={row['risk_level']:<13} "
                           f"cited={row['knowledge_cited'] or 'none'} "
                           f"unc={row['uncertainties']}")
                for problem in row["problems"]:
                    out.append(f"        PROBLEM: {problem}")

    out.append(_section("8. FAILURE CASES"))
    failures = _failures(result)
    if not failures:
        out.append("  none recorded in the deterministic sections")
    for failure in failures:
        out.append(f"  - {failure}")

    out.append(_section("9. RECOMMENDED DEFAULTS  (evidence only -- nothing changed)"))
    rec = result["recommendations"]
    if not rec.get("available"):
        out.append(f"  unavailable: {rec.get('reason')}")
    else:
        for name in ("per_query_top_k", "final_top_k", "max_per_document",
                     "min_similarity", "max_items", "max_chars", "max_total_tokens",
                     "max_flows"):
            entry = rec[name]
            out.append(f"  {name:<18} {str(entry['value']):<8} {entry['why']}")

    out.append(_section("10. LIMITATIONS"))
    out.extend(f"  - {line}" for line in LIMITATIONS)
    return "\n".join(out) + "\n"


def _render_retrieval(retrieval: dict[str, Any]) -> list[str]:
    out = [_section("3. RETRIEVAL  (shipped defaults)")]
    out.append(f"  config: {retrieval['config']}")
    out.append("\n  Per case (document level, K=8):")
    for row in retrieval["cases"]:
        if "skipped" in row:
            out.append(f"    {row['case']:<34} skipped: {row['skipped']}")
            continue
        headline = next(m for m in row["document_metrics"] if m["k"] == HEADLINE_K)
        out.append(f"    {row['case']:<34} {row['retrieved_chunks']} chunk(s)  "
                   f"recall={_show(headline['recall'], 2)} "
                   f"prec={_show(headline['precision'], 2)} "
                   f"rr={_show(headline['reciprocal_rank'], 2)}")
        if row["relevant_missed"]:
            out.append(f"        missed: {row['relevant_missed']}")
        if row["irrelevant_retrieved"]:
            out.append(f"        irrelevant retrieved: {row['irrelevant_retrieved']}")

    out.append("\n4. RECALL@K / PRECISION@K / HIT@K / MRR")
    out.append("\n  Document level:")
    for summary in retrieval["document_summary"]:
        out.append("    " + MetricSummary(**summary).row())
    out.append("\n  Section (chunk) level:")
    for summary in retrieval["section_summary"]:
        out.append("    " + MetricSummary(**summary).row())
    return out


def _render_budget(budget: dict[str, Any]) -> list[str]:
    out = [_section("5. CONTEXT BUDGET SWEEP")]
    header = ("      value      supplied excluded  est.tok  prompt  retained lost  "
              "recall")

    for name, key in (("max_items", "max_items"), ("max_chars", "max_chars"),
                      ("max_total_tokens", "max_total_tokens")):
        out.append(f"\n  {name} (other limits released):")
        out.append(header)
        for row in budget[name]:
            out.append(f"      {str(row[key]):<10} {row['supplied_total']:>8} "
                       f"{row['excluded_total']:>8} {row['mean_estimated_tokens']:>8} "
                       f"{row['mean_prompt_chars']:>7.0f} "
                       f"{row['relevant_retained_total']:>9} "
                       f"{row['relevant_lost_total']:>4} "
                       f"{_show(row['supplied_recall'], 2):>7}")

    out.append("\n  Combined configurations:")
    out.append(header)
    for row in budget["combined"]:
        out.append(f"      {row['label']:<26} {row['supplied_total']:>3} "
                   f"{row['excluded_total']:>3} {row['mean_estimated_tokens']:>8} "
                   f"{row['mean_prompt_chars']:>7.0f} "
                   f"{row['relevant_retained_total']:>4} {row['relevant_lost_total']:>4} "
                   f"{_show(row['supplied_recall'], 2):>7}")
    return out


def _render_thresholds(thresholds: dict[str, Any]) -> list[str]:
    out = [_section("6. MIN_SIMILARITY AND RETRIEVAL SHAPE")]
    out.append("\n  min_similarity:")
    out.append("      value   chunks irrelevant empty  recall  prec   MRR   retained")
    for row in thresholds["min_similarity"]:
        out.append(f"      {str(row['min_similarity']):<7} {row['retrieved_chunks']:>6} "
                   f"{row['irrelevant_documents']:>10} {row['empty_cases']:>5}  "
                   f"{_show(row['recall'], 2):>6} {_show(row['precision'], 2):>5} "
                   f"{_show(row['mrr'], 2):>5} {row['relevant_retained_after_budget']:>9}")

    for name in ("per_query_top_k", "final_top_k", "max_per_document"):
        out.append(f"\n  {name}:")
        out.append("      value   chunks irrelevant  recall  prec   MRR")
        for row in thresholds[name]:
            out.append(f"      {str(row[name]):<7} {row['retrieved_chunks']:>6} "
                       f"{row['irrelevant_documents']:>10}  {_show(row['recall'], 2):>6} "
                       f"{_show(row['precision'], 2):>5} {_show(row['mrr'], 2):>5}")
    return out


def _failures(result: dict[str, Any]) -> list[str]:
    """Everything the deterministic sections judged wrong."""
    failures: list[str] = []
    for row in result["signals"]["cases"]:
        if "skipped" in row:
            continue
        for missing in row["missing"]:
            failures.append(f"{row['case']}: expected signal {missing!r} did not fire")
        for false in row["false_detections"]:
            failures.append(f"{row['case']}: forbidden signal {false!r} fired")

    retrieval = result.get("retrieval")
    if retrieval:
        for row in retrieval["cases"]:
            if "skipped" in row:
                continue
            for missed in row["relevant_missed"]:
                failures.append(f"{row['case']}: relevant document {missed!r} was never "
                                "retrieved")
            for wrong in row["irrelevant_retrieved"]:
                failures.append(f"{row['case']}: irrelevant document {wrong!r} was "
                                "retrieved")

    live = result.get("live") or {}
    for row in live.get("cases", []):
        for problem in row.get("problems", []):
            failures.append(f"{row['case']} (live): {problem}")
    return failures


LIMITATIONS: tuple[str, ...] = (
    "Eight cases and six documents. These numbers describe this dataset, not a "
    "population; a recall of 1.00 here means the labels were found, not that "
    "retrieval is solved.",
    "Relevance labels are binary and hand-written by one author. There is no "
    "second annotator and no adjudication, so systematic bias in the labels is "
    "invisible to every metric here.",
    "Seven of eight captures are synthetic. They contain the patterns they were "
    "built to contain, which makes signal evaluation exact and makes retrieval "
    "evaluation easier than reality.",
    "The observed-fact check is numeric substring matching against the capture "
    "JSON. It catches a figure imported from reference knowledge; it cannot "
    "judge whether a sentence is a fair reading of the data.",
    "Token counts are estimates at 3.5 characters per token, not a tokenizer.",
    "The live pass is one sample per case. LLM output varies between runs even "
    "at a fixed temperature, so a single clean run is weak evidence and a "
    "single problem may not reproduce.",
    "min_similarity is swept against this dataset only. A threshold that loses "
    "nothing here may discard useful knowledge on a capture unlike these.",
    "No latency or cost measurement. Retrieval is sub-millisecond at this "
    "corpus size and was not the question.",
)


# ===========================================================================
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG + AI analysis pipeline.")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON instead of the report")
    parser.add_argument("--live", action="store_true",
                        help="also run a small live-LLM grounding pass (uses quota)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Corpus loading and index construction print progress; keep the report clean.
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = run(live=args.live)

    if args.json:
        print(_redact(json.dumps(result, indent=2, sort_keys=True, default=str)))
    else:
        print(_redact(render(result)))

    # A failing evaluation is information, not a broken program: exit 0 unless
    # the harness itself could not run.
    return 0


if __name__ == "__main__":
    sys.exit(main())
