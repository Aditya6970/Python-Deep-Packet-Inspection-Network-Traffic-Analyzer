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

Sections
--------
1 dataset, 2 signal extraction, 2b request composition, 3-4 retrieval metrics,
4b before/after by one change at a time, 4c named candidates priced in tokens,
5 context budget sweep, 6 threshold sweeps, 7 live grounding, 8 failures,
9 recommended defaults, 10 limitations.

Sections 1, 2, 2b, 8, 9 and 10 need no embedding model and run anywhere; the
rest state that they were skipped, and why, rather than being estimated.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Sequence

# Import the package so its console-encoding fix is applied before any output.
import dpi  # noqa: F401
from ai.schemas import CaptureReport
from evaluation.candidates import (
    CANDIDATES,
    OBSERVED_FAILING_PROMPT_TOKENS,
    Candidate,
    CandidateAccount,
    CaseAccount,
    rank,
)
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
# Phase 4d -- what a request is actually made of
# ===========================================================================
#: Knowledge budgets priced against every capture, in estimated tokens.
#:
#: The shipped ceiling and the one every step 10 candidate proposes, so the
#: difference between them can be read directly rather than inferred.
_BUDGET_CEILINGS: tuple[int, ...] = (900, 1200)


def evaluate_request_size() -> dict[str, Any]:
    """Price the whole request, per case, with no embedding model involved.

    This section exists because the context budget bounds the knowledge block
    and the live failures are about the *request*, and those are not the same
    thing by a wide margin. Everything measured here is independent of which
    chunks retrieval happens to pick: the capture JSON is whatever the DPI
    engine produced, the instructions are fixed, and the knowledge block is
    bounded by its ceiling whatever is in it. So this runs on any machine,
    including one where the embedding model will not load, and it is the part
    of the evaluation that speaks to the 413s.

    An upper bound, stated as one: each row is the capture-only request plus
    the *full* knowledge ceiling plus whatever the provider is sent alongside.
    A run that supplies less knowledge than its budget allows produces a
    smaller request than the figure here, never a larger one.
    """
    from ai.llm_client import estimate_request_tokens, response_format_tokens
    from ai.prompts import build_messages
    from ai.config import AIConfig
    from ai.providers import Provider
    from ai.schemas import AnalysisResult

    alongside = {
        provider.value: response_format_tokens(AIConfig(provider=provider),
                                               AnalysisResult)
        for provider in Provider
    }
    groq_extra = alongside[Provider.GROQ.value]

    rows: list[dict[str, Any]] = []
    for case in CASES:
        report = case.report()
        if report is None:
            rows.append({"case": case.case_id, "skipped": "capture unavailable"})
            continue
        messages = build_messages(report, None, None)
        capture_only = estimate_request_tokens(messages)
        by_format = _capture_only_by_format(report)
        rows.append({
            "case": case.case_id,
            "capture_by_format": by_format,
            "flows": len(report.flows),
            "capture_only_tokens": capture_only,
            "capture_only_chars": sum(len(m["content"]) for m in messages),
            "with_budget": {str(ceiling): capture_only + ceiling
                            for ceiling in _BUDGET_CEILINGS},
            "with_budget_and_schema": {
                str(ceiling): capture_only + ceiling + groq_extra
                for ceiling in _BUDGET_CEILINGS},
        })

    scored = [row for row in rows if "skipped" not in row]
    shares = [
        {str(ceiling): round(ceiling / (row["capture_only_tokens"] + ceiling), 4)
         for ceiling in _BUDGET_CEILINGS}
        for row in scored
    ]
    return {
        "response_format_tokens": alongside,
        "budget_ceilings": list(_BUDGET_CEILINGS),
        "observed_failing_prompt_tokens": OBSERVED_FAILING_PROMPT_TOKENS,
        "cases": rows,
        "largest_capture_only": max((row["capture_only_tokens"] for row in scored),
                                    default=0),
        "knowledge_share_range": {
            str(ceiling): [
                min((share[str(ceiling)] for share in shares), default=0.0),
                max((share[str(ceiling)] for share in shares), default=0.0),
            ] for ceiling in _BUDGET_CEILINGS
        },
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
            # Phase B diagnostics: what was asked, what came back, and why it
            # was ranked where it was.  Recorded for every case rather than
            # only the interesting ones, so a future regression is visible in
            # the same place a current one is.
            "query_detail": [
                {"label": query.label,
                 "signal_id": query.signal_id,
                 "topic": query.text.splitlines()[0]}
                for query in retrieval.queries
            ],
            "chunk_detail": [
                {"rank": chunk.rank,
                 "document": chunk.document_id,
                 "section": chunk.section,
                 "similarity": round(chunk.similarity, 4),
                 "matched_by": list(chunk.matched_query_labels),
                 "best_query": max(chunk.per_query_similarity,
                                   key=lambda label: (chunk.per_query_similarity[label],
                                                      label)),
                 "compatibility": chunk.compatibility.value,
                 "tier": chunk.affinity_tier,
                 "relevant": chunk.document_id in case.relevant_documents,
                 "labelled_irrelevant": chunk.document_id in case.irrelevant_documents,
                 "why": chunk.affinity_note}
                for chunk in retrieval.chunks
            ],
            "notes": list(retrieval.notes),
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
# Phase 3b -- before / after, one change at a time
# ===========================================================================
#: The configurations compared, in the order they are reported.
#:
#: ``baseline`` is the configuration measured in step 8, reproduced exactly.
#: Each ``+`` row turns on exactly one change, so a difference in the totals can
#: be attributed rather than guessed at, and ``shipped`` is what the defaults
#: now produce.  Nothing here is a tuning grid -- it is a controlled comparison
#: of named alternatives, and every row is run against the same index.
#:
#: The ``?`` rows are **candidates that are not shipped**.  They exist because
#: the step 8 sweep suggested them and the argument for them is not settled:
#: with six documents and eight slots, one chunk per document puts the entire
#: corpus in every result, which reads as diversity and can equally well read
#: as "everything, including the notes that do not apply".  Measuring them here
#: costs one more pass over an index that is already built, and turns a
#: plausible-sounding default into a decision someone can check.
def _variant_configs() -> list[tuple[str, Any, str]]:
    from ai.rag.affinity import AffinityMode
    from ai.rag.retrieval import RetrievalConfig

    return [
        ("baseline", RetrievalConfig(affinity=AffinityMode.OFF,
                                     query_style="security"),
         "step 8 defaults: similarity only"),
        ("+query wording", RetrievalConfig(affinity=AffinityMode.OFF,
                                           query_style="topical"),
         "neutral lead-in, nothing else"),
        ("+compatibility", RetrievalConfig(affinity=AffinityMode.RANK,
                                           query_style="security"),
         "applies_to tiering, nothing else"),
        ("shipped", RetrievalConfig(),
         "both together -- the current defaults"),
        ("?diversity", RetrievalConfig(max_per_document=1),
         "CANDIDATE, not shipped: shipped plus one chunk per document"),
        ("?tight budget", RetrievalConfig(max_per_document=1, final_top_k=5),
         "CANDIDATE, not shipped: retrieve about what the budget can carry"),
    ]


def _variant_row(label: str, why: str, config: Any, store: Any, embedder: Any,
                 k_values: Sequence[int]) -> dict[str, Any]:
    """Score one configuration over every case, retrieved *and* supplied.

    Two levels, because they answer different questions and only one of them is
    about the model's input:

    * **retrieved** -- everything the ranking returned. Useful for seeing what
      the corpus offered.
    * **supplied** -- what survived the shipped context budget and actually
      reached the prompt. This is the level a ranking change is *for*: with six
      documents and eight slots, demoting a note cannot remove it from the
      retrieved set, but it very much can keep it out of the four excerpts the
      budget can afford.

    The budget is held fixed at the shipped configuration across every variant,
    so the comparison is "better evidence for the same tokens" rather than
    "more tokens".
    """
    from ai.rag.context import KnowledgeContextConfig, build_knowledge_context

    budget = KnowledgeContextConfig()
    document_rows: list[RetrievalMetrics] = []
    section_rows: list[RetrievalMetrics] = []
    supplied_rows: list[RetrievalMetrics] = []
    per_case: list[dict[str, Any]] = []
    missed: list[str] = []
    wrong: list[str] = []
    supplied_missed: list[str] = []
    supplied_wrong: list[str] = []
    tokens: list[int] = []
    scored = 0

    for case in CASES:
        report = case.report()
        if report is None:
            continue
        scored += 1
        retrieval = _retrieve(case, report, store, embedder, config)
        documents = _document_ranking(retrieval.chunks)
        for k in k_values:
            document_rows.append(score_ranking(documents, case.relevant_documents, k,
                                               "document", case.irrelevant_documents))
            section_rows.append(score_ranking(_section_ranking(retrieval.chunks),
                                              case.relevant_sections, k, "chunk",
                                              dedupe=False))

        context = build_knowledge_context(retrieval, budget)
        supplied = [item.document_id for item in context.items]
        tokens.append(context.estimated_tokens)
        supplied_rows.append(score_ranking(supplied, case.relevant_documents,
                                           budget.max_items, "document",
                                           case.irrelevant_documents))

        case_missed = sorted(case.relevant_documents - set(documents))
        case_wrong = sorted(case.irrelevant_documents & set(documents))
        case_supplied_missed = sorted(case.relevant_documents - set(supplied))
        case_supplied_wrong = sorted(case.irrelevant_documents & set(supplied))
        missed.extend(f"{case.case_id}:{doc}" for doc in case_missed)
        wrong.extend(f"{case.case_id}:{doc}" for doc in case_wrong)
        supplied_missed.extend(f"{case.case_id}:{doc}" for doc in case_supplied_missed)
        supplied_wrong.extend(f"{case.case_id}:{doc}" for doc in case_supplied_wrong)
        per_case.append({
            "case": case.case_id,
            "documents": list(dict.fromkeys(documents)),
            "supplied": supplied,
            "missed": case_missed,
            "irrelevant": case_wrong,
            "supplied_missed": case_supplied_missed,
            "supplied_irrelevant": case_supplied_wrong,
            "estimated_tokens": context.estimated_tokens,
        })

    supplied_summary = aggregate(supplied_rows, budget.max_items, "document").model_dump()
    return {
        "label": label,
        "why": why,
        "config": config.as_dict(),
        "budget": budget.describe(),
        "cases_scored": scored,
        "documents": {str(k): aggregate(document_rows, k, "document").model_dump()
                      for k in k_values},
        "sections": {str(k): aggregate(section_rows, k, "chunk").model_dump()
                     for k in k_values},
        "supplied": supplied_summary,
        "mean_estimated_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
        "missed": missed,
        "irrelevant": wrong,
        "supplied_missed": supplied_missed,
        "supplied_irrelevant": supplied_wrong,
        # The four numbers the decision rule reads, gathered in one place so the
        # rule cannot quietly be applied to a different set than the one shown.
        "headline": {
            "recall": supplied_summary["recall"],
            "precision": supplied_summary["precision"],
            "irrelevant": len(supplied_wrong),
            "missed": len(supplied_missed),
        },
        "per_case": per_case,
    }


def _decide(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Apply the accept/reject rule to a before/after pair.

    Judged on the **supplied** set -- the excerpts that reached the prompt under
    the shipped budget -- because that is the only knowledge the model ever saw.
    A note demoted to the bottom of an eight-slot result on a six-document
    corpus is still "retrieved"; what changed is whether it was put in front of
    the model, and that is the thing worth measuring.

    The rule is fixed in advance and is deliberately asymmetric: a change that
    trades precision away for recall is rejected outright, while a change that
    trades recall away for precision is rejected unless the loss is small. A
    retriever that supplies the wrong document confidently is worse than one
    that supplies one fewer right document, because a person reading the output
    cannot tell the first case from a correct answer.
    """
    first = before.get("headline", {})
    second = after.get("headline", {})
    recall_before, recall_after = first.get("recall"), second.get("recall")
    prec_before, prec_after = first.get("precision"), second.get("precision")

    if None in (recall_before, recall_after, prec_before, prec_after):
        return {"verdict": "undecided",
                "why": "a headline metric was undefined; nothing to compare"}

    bad_before, bad_after = first.get("irrelevant", 0), second.get("irrelevant", 0)
    missed_before, missed_after = first.get("missed", 0), second.get("missed", 0)
    recall_delta = recall_after - recall_before
    precision_delta = prec_after - prec_before
    reasons = [
        f"supplied recall {recall_before:.3f} -> {recall_after:.3f} ({recall_delta:+.3f})",
        f"supplied precision {prec_before:.3f} -> {prec_after:.3f} "
        f"({precision_delta:+.3f})",
        f"irrelevant documents supplied {bad_before} -> {bad_after}",
        f"relevant documents not supplied {missed_before} -> {missed_after}",
    ]

    # "Materially" is 0.05 of recall -- with eight cases that is closer to
    # half a case than a whole one, so it cannot be reached by rounding.
    if precision_delta < 0 or bad_after > bad_before:
        verdict, why = "reject", ("precision fell or more irrelevant documents "
                                  "were supplied")
    elif recall_delta < -0.05:
        verdict, why = "reject", "recall fell materially"
    elif recall_delta <= 0 and precision_delta <= 0 and bad_after == bad_before:
        verdict, why = "reject", "nothing measurably improved"
    else:
        verdict, why = "accept", ("precision and irrelevant-supply counts improved "
                                  "without material recall loss")

    return {"verdict": verdict, "why": why, "evidence": reasons,
            "recall_delta": round(recall_delta, 4),
            "precision_delta": round(precision_delta, 4),
            "irrelevant_delta": bad_after - bad_before,
            "missed_delta": missed_after - missed_before}


def evaluate_variants(store: Any, embedder: Any,
                      k_values: Sequence[int] = DEFAULT_K_VALUES) -> dict[str, Any]:
    """Score every named configuration, then judge shipped against baseline."""
    rows = [_variant_row(label, why, config, store, embedder, k_values)
            for label, config, why in _variant_configs()]
    by_label = {row["label"]: row for row in rows}
    return {
        "k_values": list(k_values),
        "variants": rows,
        "decision": _decide(by_label["baseline"], by_label["shipped"]),
    }


# ===========================================================================
# Phase 4c -- named candidates, priced in tokens
# ===========================================================================
def _prompt_sizes(report: CaptureReport, knowledge: str | None) -> tuple[int, int]:
    """``(characters, estimated tokens)`` of the whole request for one case.

    The whole request, not the knowledge block: system prompt, framing, capture
    JSON and knowledge together, measured through the same
    :func:`~ai.prompts.build_messages` the analyzer calls, so the number is the
    thing that gets sent rather than a model of it.

    ``schema=None`` on purpose. Groq is a ``JSON_SCHEMA`` provider: the schema
    travels in ``response_format``, not in the prompt. It is still metered, and
    :func:`provider_schema_tokens` reports it separately rather than being
    folded in here, where it would quietly inflate every provider's number.
    """
    from ai.llm_client import estimate_request_tokens
    from ai.prompts import build_messages

    messages = build_messages(report, None, knowledge)
    characters = sum(len(message["content"]) for message in messages)
    return characters, estimate_request_tokens(messages)


def _capture_only_by_format(report: CaptureReport) -> dict[str, int]:
    """Capture-only prompt size under each rendering, for the before/after."""
    from ai.llm_client import estimate_request_tokens
    from ai.prompts import CAPTURE_FORMATS, build_messages

    return {name: estimate_request_tokens(build_messages(report, None, None, name))
            for name in CAPTURE_FORMATS}


def provider_schema_tokens() -> int:
    """Estimated size of the JSON Schema Groq is sent alongside the prompt.

    Measured, not assumed: a request to a ``JSON_SCHEMA`` provider carries this
    in addition to everything :func:`_prompt_sizes` counts, so a token budget
    that ignores it is short by exactly this much.
    """
    from ai.llm_client import estimate_request_tokens
    from ai.providers import to_strict_json_schema
    from ai.schemas import AnalysisResult

    body = json.dumps(to_strict_json_schema(AnalysisResult.model_json_schema()),
                      sort_keys=True)
    return estimate_request_tokens([{"role": "system", "content": body}])


def _account_for(candidate: Candidate, store: Any, embedder: Any,
                 k: int = HEADLINE_K) -> CandidateAccount:
    """Run every case under one candidate and total up what it did and cost."""
    from ai.rag.context import KnowledgeContextConfig, build_knowledge_context
    from ai.rag.retrieval import RetrievalConfig

    retrieval_config = RetrievalConfig(**dict(candidate.retrieval))
    budget = KnowledgeContextConfig(**dict(candidate.budget))
    # Counted once, not per case: it is the same schema every time, and it is
    # metered by the provider exactly like the prompt.
    alongside = provider_schema_tokens()

    accounts: list[CaseAccount] = []
    retrieval_rows: list[RetrievalMetrics] = []
    supplied_rows: list[RetrievalMetrics] = []

    for case in CASES:
        report = case.report()
        if report is None:
            continue

        retrieval = _retrieve(case, report, store, embedder, retrieval_config)
        context = build_knowledge_context(retrieval, budget)
        documents = tuple(dict.fromkeys(_document_ranking(retrieval.chunks)))
        supplied = tuple(dict.fromkeys(item.document_id for item in context.items))

        knowledge_text = context.text or None
        prompt_chars, prompt_tokens = _prompt_sizes(report, knowledge_text)
        _, capture_only = _prompt_sizes(report, None)

        retrieval_rows.append(score_ranking(list(documents), case.relevant_documents,
                                            k, "document", case.irrelevant_documents))
        # Scored at the number of excerpts this budget may supply, so precision
        # is not penalised for slots the configuration never offered.
        supplied_rows.append(score_ranking(list(supplied), case.relevant_documents,
                                           budget.max_items, "document",
                                           case.irrelevant_documents))

        accounts.append(CaseAccount(
            case_id=case.case_id,
            retrieved_documents=documents,
            supplied_documents=supplied,
            relevant=frozenset(case.relevant_documents),
            irrelevant=frozenset(case.irrelevant_documents),
            knowledge_chars=context.total_chars,
            knowledge_tokens=context.estimated_tokens,
            prompt_chars=prompt_chars,
            prompt_tokens=prompt_tokens,
            capture_only_tokens=capture_only,
            excluded_by_budget=context.dropped_items,
            alongside_tokens=alongside,
        ))

    return CandidateAccount(
        candidate=candidate,
        cases=tuple(accounts),
        retrieval_metrics=aggregate(retrieval_rows, k, "document").model_dump(),
        supplied_metrics=aggregate(supplied_rows, budget.max_items,
                                   "document").model_dump(),
    )


#: Cases the step 9 analysis singled out, watched individually here.
WATCHED_CASES: tuple[str, ...] = (
    "normal-cdn-multi-host", "unknown-application", "real-capture-benign",
    "knowledge-conflicts-observation", "dns-tunneling", "two-documents-relevant",
)


def evaluate_candidates(store: Any, embedder: Any) -> dict[str, Any]:
    """Score every named candidate and apply the selection rule."""
    accounts = [_account_for(candidate, store, embedder) for candidate in CANDIDATES]
    selection = rank(accounts)

    watched: dict[str, dict[str, Any]] = {}
    for account in accounts:
        for case in account.cases:
            if case.case_id in WATCHED_CASES:
                watched.setdefault(case.case_id, {})[account.candidate.name] = {
                    "supplied": list(case.supplied_documents),
                    "never_retrieved": list(case.never_retrieved),
                    "lost_before_prompt": list(case.lost_before_prompt),
                    "irrelevant_supplied": list(case.irrelevant_supplied),
                    "prompt_tokens": case.prompt_tokens,
                    "knowledge_tokens": case.knowledge_tokens,
                }

    return {
        "observed_failing_prompt_tokens": OBSERVED_FAILING_PROMPT_TOKENS,
        "provider_schema_tokens": provider_schema_tokens(),
        "candidates": [account.as_dict() for account in accounts],
        "selection": selection,
        "watched_cases": watched,
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
        "request_size": evaluate_request_size(),
        "retrieval": None,
        "variants": None,
        "candidates": None,
        "budget": None,
        "thresholds": None,
        "live": None,
        "index": {"available": False, "reason": ""},
    }

    try:
        store, embedder, chunk_count = build_index()
    except IndexUnavailable as exc:
        result["index"] = {"available": False, "reason": str(exc)}
        result["variants"] = {"skipped": f"the knowledge index is unavailable: {exc}"}
        result["candidates"] = {"skipped": f"the knowledge index is unavailable: {exc}"}
        result["recommendations"] = recommend(None, None, None)
        # The live pass needs the same index, so it cannot run either.  Saying
        # so is better than leaving the section null and letting a reader guess.
        result["live"] = {"skipped": f"the knowledge index is unavailable: {exc}"}
        return result

    result["index"] = {"available": True, "chunks": chunk_count,
                       "dimension": store.dimension, "model": store.model_name}
    result["retrieval"] = evaluate_retrieval(store, embedder)
    result["variants"] = evaluate_variants(store, embedder)
    result["candidates"] = evaluate_candidates(store, embedder)
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

    out.extend(_render_request_size(result["request_size"]))

    index = result["index"]
    if not index["available"]:
        out.append(_section("3-6. RETRIEVAL, BUDGET AND THRESHOLD SWEEPS"))
        out.append(f"  SKIPPED: {index['reason']}")
        out.append("  These sections need the real embedding model; they are never")
        out.append("  estimated. Install requirements-rag.txt and re-run.")
    else:
        out.extend(_render_retrieval(result["retrieval"]))
        out.extend(_render_variants(result["variants"]))
        out.extend(_render_candidates(result["candidates"]))
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


def _render_variants(variants: dict[str, Any]) -> list[str]:
    out = [_section("4b. BEFORE / AFTER  (same index, same context budget)")]
    if "skipped" in variants:
        out.append(f"  SKIPPED: {variants['skipped']}")
        return out

    out.append("  'supplied' is what the shipped context budget actually put in the")
    out.append("  prompt; 'retrieved' is everything the ranking returned. A demotion")
    out.append("  shows up in the first and usually not in the second, because on a")
    out.append("  six-document corpus eight slots hold nearly all of it either way.")
    out.append("")
    out.append("      configuration      | supplied: recall  prec  irrel miss  ~tok "
               "| retrieved: recall@8 prec@3 irrel")
    for row in variants["variants"]:
        sup = row["supplied"]
        docs8, docs3 = row["documents"]["8"], row["documents"]["3"]
        out.append(f"      {row['label']:<18} |           "
                   f"{_show(sup['recall'], 3):>6} {_show(sup['precision'], 3):>5} "
                   f"{row['headline']['irrelevant']:>5} "
                   f"{row['headline']['missed']:>4} "
                   f"{row['mean_estimated_tokens']:>5.0f} "
                   f"|            {_show(docs8['recall'], 3):>7} "
                   f"{_show(docs3['precision'], 3):>6} {len(row['irrelevant']):>5}")
        out.append(f"        {row['why']}")

    decision = variants["decision"]
    out.append(f"\n  Decision rule (shipped vs baseline, on the supplied set): "
               f"{decision['verdict'].upper()}")
    out.append(f"    {decision['why']}")
    for line in decision.get("evidence", []):
        out.append(f"    {line}")

    out.append("\n  Under the shipped configuration, cases where the supplied set is "
               "still wrong:")
    shipped = next(row for row in variants["variants"] if row["label"] == "shipped")
    clean = True
    for case in shipped["per_case"]:
        if case["supplied_missed"] or case["supplied_irrelevant"]:
            clean = False
            out.append(f"    {case['case']:<34} not supplied={case['supplied_missed']} "
                       f"irrelevant supplied={case['supplied_irrelevant']}")
    if clean:
        out.append("    none")
    return out


def _render_request_size(sizes: dict[str, Any]) -> list[str]:
    out = [_section("2b. WHAT A REQUEST IS MADE OF  (no embedding model needed)")]
    groq = sizes["response_format_tokens"].get("groq", 0)
    failing = sizes["observed_failing_prompt_tokens"]
    out.append(f"  Estimated tokens. A JSON_SCHEMA provider is sent the response schema")
    out.append(f"  as well as the prompt: ~{groq} tokens that no context budget counts.")
    out.append(f"  A request of ~{failing} tokens has been observed to fail live.")
    first = next((row for row in sizes["cases"] if "skipped" not in row), None)
    if first and "capture_by_format" in first:
        out.append("  Capture rendering (prompt v2.0 lays the flows out as a table;")
        out.append("  DPI_CAPTURE_FORMAT=json restores the previous layout):")
        for row in sizes["cases"]:
            if "skipped" in row:
                continue
            fmts = row["capture_by_format"]
            saving = 1 - fmts["table"] / fmts["json"] if fmts["json"] else 0.0
            out.append(f"      {row['case']:<34} json {fmts['json']:>5} -> "
                       f"table {fmts['table']:>5} tokens  ({saving:.0%} smaller)")
    out.append("")
    ceilings = sizes["budget_ceilings"]
    header = (f"      {'case':<34} {'flows':>5} {'capture':>8}"
              + "".join(f"{'+' + str(c):>8}" for c in ceilings)
              + "".join(f"{'+' + str(c) + '+sch':>12}" for c in ceilings))
    out.append(header)
    for row in sizes["cases"]:
        if "skipped" in row:
            out.append(f"      {row['case']:<34} skipped: {row['skipped']}")
            continue
        line = (f"      {row['case']:<34} {row['flows']:>5} "
                f"{row['capture_only_tokens']:>8}")
        line += "".join(f"{row['with_budget'][str(c)]:>8}" for c in ceilings)
        for c in ceilings:
            total = row["with_budget_and_schema"][str(c)]
            line += f"{total:>11}{'!' if total >= failing else ' '}"
        out.append(line)

    out.append("")
    for ceiling in ceilings:
        low, high = sizes["knowledge_share_range"][str(ceiling)]
        out.append(f"  At a {ceiling}-token ceiling the knowledge block is "
                   f"{low * 100:.0f}%-{high * 100:.0f}% of the prompt;")
        out.append("  the capture is the rest.")
    out.append(f"  Largest capture-only request: "
               f"~{sizes['largest_capture_only']} tokens.")
    out.append("  Reading: the capture is still the larger term, so the knowledge")
    out.append("  budget alone cannot decide whether a request is sendable. Prompt")
    out.append("  v2.0 attacked the large term directly by laying the flows out as a")
    out.append("  table instead of as pretty-printed JSON -- same flows, same fields,")
    out.append("  same values, roughly half the tokens. --max-flows remains the only")
    out.append("  other lever on this term, and it is the one that discards evidence.")
    return out


def _render_candidates(candidates: dict[str, Any]) -> list[str]:
    out = [_section("4c. CANDIDATE CONFIGURATIONS, PRICED IN TOKENS")]
    if "skipped" in candidates:
        out.append(f"  SKIPPED: {candidates['skipped']}")
        return out

    failing = candidates["observed_failing_prompt_tokens"]
    out.append(f"  A prompt of ~{failing} estimated tokens has been observed to fail")
    out.append("  live (HTTP 413/429). That is one-sided evidence: it says this size")
    out.append("  fails, not that anything smaller succeeds. It flags rows; it guards")
    out.append("  nothing.")
    out.append(f"  A JSON_SCHEMA provider (Groq) is additionally sent the response")
    out.append(f"  schema, about {candidates['provider_schema_tokens']} estimated "
               "tokens, which none of the")
    out.append("  prompt figures below include.")
    out.append("")
    out.append("                 retrieval          |            supplied            | "
               "         cost")
    out.append("      name       recall  prec   MRR | recall  prec  lost irrel excl | "
               "~know  ~max prompt  know%  ev/1k")
    for row in candidates["candidates"]:
        excluded = sum(case["excluded_by_budget"] for case in row["per_case"])
        risk = " !" if row["cases_at_risk"] else "  "
        out.append(
            f"      {row['name']:<10} {_show(row['retrieval_recall'], 3):>6} "
            f"{_show(row['retrieval_precision'], 3):>5} {_show(row['mrr'], 3):>5} | "
            f"{_show(row['supplied_recall'], 3):>6} "
            f"{_show(row['supplied_precision'], 3):>5} "
            f"{row['lost_before_prompt']:>5} {row['irrelevant_supplied']:>5} "
            f"{excluded:>4} | {row['mean_knowledge_tokens']:>5.0f} "
            f"{row['max_request_tokens']:>11}{risk} "
            f"{row['mean_knowledge_share'] * 100:>5.1f}% "
            f"{_show(row['evidence_per_1k_tokens'], 3):>6}")
        marker = "  (reference only)" if row["reference_only"] else ""
        out.append(f"        {row['description']}{marker}")
        out.append(f"        {row['configuration']}")

    out.append("\n  lost  = relevant, retrieved, then dropped by the budget")
    out.append("  irrel = labelled-irrelevant documents that reached the prompt")
    out.append("  excl  = excerpts retrieval found and the budget could not afford")
    out.append("  know% = share of the request the knowledge block accounts for")
    out.append("  ev/1k = supplied recall per 1000 tokens of the LARGEST request")
    out.append("  max request = largest prompt PLUS what the provider is sent "
               "alongside it")
    out.append("  !     = at least one case at or above a size observed to fail")

    selection = candidates["selection"]
    if not selection.get("size_axis_informative", True):
        out.append(
            f"\n  NOTE: the baseline's own largest request is "
            f"~{selection.get('baseline_max_request_tokens')} tokens, already at or "
            f"above the ~{failing} observed to fail. Request size therefore does not")
        out.append("  separate these candidates, and none of them is thereby safe to")
        out.append("  send. The term that would shrink a request is the capture "
                   "(--max-flows),")
        out.append("  not the knowledge budget.")
    out.append(f"\n  Selection rule (against {selection['baseline']}, "
               f"ev/1k {_show(selection['baseline_score'], 3)}):")
    for verdict in selection["verdicts"]:
        mark = "ADMISSIBLE" if verdict["admissible"] else "rejected  "
        out.append(f"    {mark}  {verdict['name']:<10} ev/1k="
                   f"{_show(verdict['score'], 3)}")
        for reason in verdict["reasons"]:
            out.append(f"        {reason}")
    recommended = selection["recommended"]
    out.append(f"\n  RECOMMENDED CANDIDATE: {recommended or 'none -- keep the shipped defaults'}")
    out.append(f"  {selection['note']}")

    out.append("\n  Watched cases, by candidate:")
    for case_id in WATCHED_CASES:
        rows = candidates["watched_cases"].get(case_id)
        if not rows:
            out.append(f"    {case_id:<34} (capture unavailable)")
            continue
        out.append(f"    {case_id}")
        for name in [row["name"] for row in candidates["candidates"]]:
            entry = rows.get(name)
            if entry is None:
                continue
            problems = []
            if entry["never_retrieved"]:
                problems.append(f"never retrieved {entry['never_retrieved']}")
            if entry["lost_before_prompt"]:
                problems.append(f"lost to budget {entry['lost_before_prompt']}")
            if entry["irrelevant_supplied"]:
                problems.append(f"irrelevant {entry['irrelevant_supplied']}")
            out.append(f"      {name:<10} {len(entry['supplied'])} supplied, "
                       f"~{entry['prompt_tokens']} prompt tokens"
                       + (f"  -- {'; '.join(problems)}" if problems else "  -- clean"))
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
    "OBSERVED_FAILING_PROMPT_TOKENS is one-sided evidence. A request of that "
    "size failed live; nothing here establishes that a smaller one succeeds, "
    "and the real limit is a property of the provider account rather than of "
    "this code. It flags rows in a report and guards nothing.",
    "Token figures throughout are the 3.5-characters-per-token estimate, which "
    "runs high against a real tokenizer. That is the safe direction for a size "
    "warning and the wrong direction for a size guarantee: a candidate shown as "
    "fitting is not thereby proven to fit.",
    "The candidate table prices the whole request but scores knowledge only. "
    "Nothing in it measures whether a supplied excerpt changed the model's "
    "answer -- that is the live grounding pass, at one sample per case.",
    "The compatibility tier reads each document's hand-written applies_to list. "
    "It is only as good as that declaration: a document whose author forgot to "
    "list a signal it genuinely covers will be ranked below one that remembered, "
    "and no metric here can tell that from a correct demotion.",
    "The before/after table judges the supplied set under one fixed budget. A "
    "ranking change that only reorders excerpts already inside the budget is "
    "invisible to it, which is intended -- but it also means the table says "
    "nothing about how the ranking would behave at a larger budget.",
    "No latency or cost measurement. Retrieval is sub-millisecond at this "
    "corpus size and was not the question.",
)


# ===========================================================================
def _emit(text: str, path: str | None, json_mode: bool) -> int:
    """Deliver the report, and make the delivery itself trustworthy.

    Writing the file here rather than leaving it to the shell is not a
    convenience. ``python run_rag_evaluation.py --json > out.json`` in Windows
    PowerShell 5.1 goes through ``Out-File``, whose default encoding is
    UTF-16LE **with a byte-order mark** -- so a perfectly good JSON document
    arrives on disk as ``ff fe 7b 00 ...`` and ``json.load(open(path))`` fails
    with ``Expecting value: line 1 column 1 (char 0)``. The program was right,
    the file was wrong, and nothing in the error said so.

    ``--out`` removes the shell from the path: the file is written UTF-8, no
    BOM, ``\n`` line endings, and in JSON mode it is read back and parsed
    before this function returns. A file this reports as written has been
    proven to parse.
    """
    if path is None:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        if json_mode:
            encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
            if encoding.replace("-", "") not in ("utf8", ""):
                print(f"note: stdout is {encoding}; if you are redirecting to a "
                      "file, prefer --out PATH, which always writes UTF-8.",
                      file=sys.stderr)
        return 0

    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text + "\n")

    if json_mode:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: wrote {path} but it does not parse as JSON: {exc}",
                  file=sys.stderr)
            return 1
        print(f"wrote {path} ({os.path.getsize(path)} bytes, UTF-8, verified "
              "to parse as JSON)", file=sys.stderr)
    else:
        print(f"wrote {path} ({os.path.getsize(path)} bytes, UTF-8)",
              file=sys.stderr)
    return 0


def _failure_document(exc: BaseException) -> dict[str, Any]:
    """What JSON mode emits when the run itself fell over.

    An empty file and a traceback on a stream the user redirected away is the
    worst of both worlds: nothing to read and nothing to parse. A machine-
    readable mode that cannot report its own failure machine-readably is not
    machine-readable, so a crash produces a valid document that says so, and
    the exit status is non-zero.
    """
    return {
        "error": {
            "type": type(exc).__name__,
            "detail": _redact(str(exc)),
            "traceback": _redact(traceback.format_exc()),
        },
        "dataset": None, "signals": None, "request_size": None, "retrieval": None,
        "variants": None, "candidates": None, "budget": None, "thresholds": None,
        "live": None, "recommendations": None,
        "index": {"available": False, "reason": "the evaluation raised before finishing"},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG + AI analysis pipeline.")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON instead of the report")
    parser.add_argument("--live", action="store_true",
                        help="also run a small live-LLM grounding pass (uses quota)")
    parser.add_argument("--out", metavar="PATH", default=None,
                        help="write the output to PATH as UTF-8 (no BOM) and, in "
                             "--json mode, verify it parses. Prefer this over shell "
                             "redirection on Windows, where PowerShell's '>' writes "
                             "UTF-16 and the resulting file will not load.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Corpus loading, index construction and sentence-transformers all print
    # progress to stdout.  It is captured here and released to stderr, so that
    # stdout carries the document and nothing else -- which is what makes
    # `--json` safe to redirect.
    progress = io.StringIO()
    try:
        with redirect_stdout(progress):
            result = run(live=args.live)
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        noise = progress.getvalue()
        if noise:
            sys.stderr.write(noise)
        traceback.print_exc(file=sys.stderr)
        if args.json:
            document = _redact(json.dumps(_failure_document(exc), indent=2,
                                          sort_keys=True, default=str))
            _emit(document, args.out, json_mode=True)
        else:
            _emit(f"EVALUATION FAILED\n\n  {type(exc).__name__}: {_redact(str(exc))}\n",
                  args.out, json_mode=False)
        return 1

    noise = progress.getvalue()
    if noise:
        sys.stderr.write(noise)

    if args.json:
        document = _redact(json.dumps(result, indent=2, sort_keys=True, default=str))
        # Parse what is about to be emitted.  Cheap, and it means the contract
        # "--json prints a JSON document" is checked rather than assumed.
        json.loads(document)
        return _emit(document, args.out, json_mode=True)

    return _emit(_redact(render(result)), args.out, json_mode=False)


if __name__ == "__main__":
    sys.exit(main())
