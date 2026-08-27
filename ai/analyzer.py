"""Orchestration: DPI snapshot in, validated analysis out.

The pipeline::

    FlowSnapshot -> extract -> redact -> prompt -> LLM -> validate -> Analysis

Every stage that can fail does so by returning a value. :func:`analyze_capture`
raises nothing under normal operation: a missing key, an unreachable API, a
timeout, a refusal, or a response that references flows that do not exist all
produce an :class:`AnalysisOutcome` with ``ok = False`` and a stated reason.

That is the whole point of the design. The DPI engine has already finished its
work and written its output before this runs; nothing here can take that away.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dpi.dpi_engine import FlowSnapshot

from .config import AIConfig
from .extractor import build_capture_report
from .llm_client import FailureReason, LLMClient, create_client
from .prompts import PROMPT_VERSION, build_messages, prompt_version
from .providers import StructuredMode
from .schemas import AnalysisResult, CaptureReport

if TYPE_CHECKING:  # pragma: no cover - typing only; ai/ never needs ai/rag/
    from .rag.context import KnowledgeContext
    from .rag.pipeline import KnowledgePipeline, RAGOutcome

__all__ = ["AnalysisOutcome", "analyze_capture", "explain_failure"]


#: Human-readable explanations, so a failure tells the user what to do.
_FAILURE_HELP: dict[FailureReason, str] = {
    FailureReason.NO_API_KEY: (
        "No API key configured for the selected provider. See .env.example. "
        "DPI analysis above is unaffected."
    ),
    FailureReason.INVALID_PROVIDER: (
        "DPI_LLM_PROVIDER names an unknown provider. Valid values are: "
        "groq, ollama, openai."
    ),
    FailureReason.PROVIDER_UNAVAILABLE: (
        "Could not reach the provider endpoint. For Ollama, check the server "
        "is running (`ollama serve`) and OLLAMA_BASE_URL is correct."
    ),
    FailureReason.SDK_MISSING: (
        "The 'openai' package is not installed. Run: pip install -r requirements.txt"
    ),
    FailureReason.AUTH_FAILED: (
        "The API rejected the key. Check OPENAI_API_KEY is correct and active."
    ),
    FailureReason.RATE_LIMITED: (
        "Rate limited after retries. Wait and try again, or check your quota."
    ),
    FailureReason.TIMEOUT: (
        "The request timed out after retries. A host that is not listening at "
        "all can also surface here rather than as a refused connection — some "
        "platforms drop the packet instead of rejecting it — so check the "
        "endpoint is actually up before raising DPI_AI_TIMEOUT."
    ),
    FailureReason.API_ERROR: "The API call failed. This is usually transient.",
    FailureReason.INVALID_RESPONSE: (
        "The model's response did not match the expected schema."
    ),
    FailureReason.REFUSED: "The model declined to answer this request.",
}


def explain_failure(
    reason: FailureReason | None, config: AIConfig | None = None
) -> str:
    """Return actionable guidance for a failure reason.

    When a config is supplied the message names the actual provider and its
    setup steps, so the user is not left guessing which key is missing.
    """
    if reason is None:
        return ""

    base = _FAILURE_HELP.get(reason, "AI analysis was unavailable.")
    if config is None:
        return base

    spec = config.spec
    if reason is FailureReason.NO_API_KEY:
        return (
            f"No API key for {spec.label}. Set {spec.api_key_env} in your "
            f"environment or .env file.\n  {spec.setup_hint}\n"
            "  DPI analysis above is unaffected."
        )
    # Reachability- and endpoint-related failures all need the provider's own
    # setup steps.  TIMEOUT belongs here because "nothing is listening" is not
    # reliably distinguishable from "the server is slow": Linux refuses a dead
    # port (APIConnectionError -> PROVIDER_UNAVAILABLE) while Windows silently
    # drops it (APITimeoutError -> TIMEOUT).  The classification stays honest
    # in both cases; only the advice is made actionable.
    if reason in (FailureReason.PROVIDER_UNAVAILABLE, FailureReason.AUTH_FAILED,
                  FailureReason.API_ERROR, FailureReason.TIMEOUT):
        return f"{base}\n  Provider: {spec.label} ({config.model})\n  {spec.setup_hint}"
    return base


@dataclass(slots=True)
class AnalysisOutcome:
    """Result of one analysis attempt, successful or not."""

    report: CaptureReport
    analysis: AnalysisResult | None = None
    failure: FailureReason | None = None
    detail: str = ""
    #: Non-fatal problems found while validating a successful response.
    warnings: list[str] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    prompt_version: str = PROMPT_VERSION
    attempts: int = 0
    elapsed_seconds: float = 0.0
    #: Kept so :meth:`guidance` can name the provider and its setup steps.
    config: AIConfig | None = None

    # --- retrieval-augmented generation, all optional --------------------
    #: The reference block supplied to the model, when one was.
    knowledge: "KnowledgeContext | None" = None
    #: Why knowledge was or was not used -- a ``RAGStatus`` value, as a string
    #: so this dataclass stays importable with no RAG dependencies installed.
    rag_status: str = "disabled"
    #: Human-readable explanation of ``rag_status``.
    rag_detail: str = ""
    #: Signal types the capture produced, when signal extraction ran.
    signal_types: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.analysis is not None

    @property
    def knowledge_used(self) -> bool:
        """True only when reference knowledge actually reached the model."""
        return self.knowledge is not None and bool(self.knowledge.items)

    def knowledge_refs(self) -> tuple[str, ...]:
        """Labels the model cited, in the order it gave them."""
        if self.analysis is None:
            return ()
        return tuple(self.analysis.knowledge_refs)

    def guidance(self) -> str:
        """What the user should do about a failure."""
        return explain_failure(self.failure, self.config)


def analyze_capture(
    snapshot: FlowSnapshot,
    capture_path: str | Path,
    config: AIConfig | None = None,
    client: LLMClient | None = None,
    rag: "KnowledgePipeline | None" = None,
) -> AnalysisOutcome:
    """Analyse one capture. Never raises for an expected failure.

    ``client`` may be injected — pass a
    :class:`~ai.llm_client.FakeLLMClient` to run the full pipeline offline.

    ``rag`` may be a :class:`~ai.rag.pipeline.KnowledgePipeline`. When supplied,
    signals are extracted from the report, knowledge is retrieved for them, and
    the excerpts are added to the prompt. **The default is ``None``**: the
    library does not turn retrieval on by itself, so every existing caller
    keeps the exact behaviour it had. ``analyze_ai.py`` opts in.

    A pipeline that cannot produce knowledge — no dependencies, no model, no
    corpus, nothing matched — is not an error. The analysis proceeds without
    it, and ``rag_status`` on the outcome says which of those happened.
    """
    cfg = config or AIConfig.from_env()
    started = time.monotonic()

    # --- deterministic stages: always run, never need a network ------------
    report = build_capture_report(snapshot, capture_path, cfg)

    # --- optional retrieval: deterministic, local, never fatal -------------
    rag_outcome: "RAGOutcome | None" = None
    knowledge_text: str | None = None
    if rag is not None:
        rag_outcome = rag.build_context(report)
        knowledge_text = rag_outcome.knowledge_text()

    # The provider is chosen here and nowhere else; this function never names
    # one.  An unrecognised DPI_LLM_PROVIDER is refused rather than silently
    # replaced with a working provider.
    llm = client if client is not None else create_client(cfg)

    if not llm.is_available():
        if cfg.invalid_provider is not None:
            reason = FailureReason.INVALID_PROVIDER
        elif not cfg.has_api_key():
            reason = FailureReason.NO_API_KEY
        else:
            reason = FailureReason.SDK_MISSING
        return AnalysisOutcome(
            report=report,
            failure=reason,
            detail=explain_failure(reason, cfg),
            provider=cfg.provider.value,
            model=cfg.model,
            config=cfg,
            elapsed_seconds=time.monotonic() - started,
            **_rag_fields(rag_outcome),
        )

    # --- non-deterministic stage ------------------------------------------
    # Providers that cannot constrain generation themselves are given the
    # schema in the prompt instead.  Either way the response is validated
    # against the same AnalysisResult below.
    schema = (
        AnalysisResult.model_json_schema()
        if cfg.spec.structured_mode is StructuredMode.JSON_OBJECT
        else None
    )
    messages = build_messages(report, schema, knowledge_text)
    result = llm.complete_structured(messages, AnalysisResult)

    if not result.ok or not isinstance(result.parsed, AnalysisResult):
        return AnalysisOutcome(
            report=report,
            failure=result.failure or FailureReason.INVALID_RESPONSE,
            detail=result.detail,
            model=result.model or cfg.model,
            provider=result.provider or cfg.provider.value,
            config=cfg,
            attempts=result.attempts,
            prompt_version=prompt_version(bool(knowledge_text)),
            elapsed_seconds=time.monotonic() - started,
            **_rag_fields(rag_outcome),
        )

    analysis = result.parsed
    supplied_refs = rag_outcome.refs if rag_outcome is not None else ()

    # --- post-validation: the mechanical hallucination checks --------------
    # An invented knowledge citation is a fabricated source, so unlike an
    # invented flow id it is fatal rather than a warning: the response claims
    # support from a document that was never supplied, and no part of it can be
    # trusted to have come from where it says.  The references are never
    # silently stripped -- that would hide the fabrication and keep the text
    # that depended on it.
    citation_problems = analysis.validate_knowledge_references(supplied_refs)
    if citation_problems:
        return AnalysisOutcome(
            report=report,
            failure=FailureReason.INVALID_RESPONSE,
            detail="; ".join(citation_problems),
            model=result.model or cfg.model,
            provider=result.provider or cfg.provider.value,
            config=cfg,
            attempts=result.attempts,
            prompt_version=prompt_version(bool(knowledge_text)),
            elapsed_seconds=time.monotonic() - started,
            **_rag_fields(rag_outcome),
        )

    # A referenced flow that does not exist is caught here rather than trusted.
    warnings = analysis.validate_flow_references(report.flow_ids())

    if not analysis.uncertainties:
        warnings.append(
            "Model reported no uncertainties; network captures are partial "
            "evidence, so treat this assessment with extra caution."
        )

    return AnalysisOutcome(
        report=report,
        analysis=analysis,
        warnings=warnings,
        model=result.model or cfg.model,
        provider=result.provider or cfg.provider.value,
        config=cfg,
        attempts=result.attempts,
        prompt_version=prompt_version(bool(knowledge_text)),
        elapsed_seconds=time.monotonic() - started,
        **_rag_fields(rag_outcome),
    )


def _rag_fields(outcome: "RAGOutcome | None") -> dict[str, object]:
    """The retrieval-related fields of an :class:`AnalysisOutcome`.

    One place, so every early return carries the same RAG state and none can
    drift into reporting "disabled" for a run that actually retrieved.
    """
    if outcome is None:
        return {}
    # The context is attached even when it supplied nothing, because an empty
    # context still records what the budget excluded.  ``knowledge_used`` keys
    # off the items, so nothing downstream mistakes "excluded" for "supplied".
    return {
        "knowledge": outcome.context,
        "rag_status": outcome.status.value,
        "rag_detail": outcome.describe(),
        "signal_types": outcome.signal_types,
    }
