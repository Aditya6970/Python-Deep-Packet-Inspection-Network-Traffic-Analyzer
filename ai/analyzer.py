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

from dpi.dpi_engine import FlowSnapshot

from .config import AIConfig
from .extractor import build_capture_report
from .llm_client import FailureReason, LLMClient, create_client
from .prompts import PROMPT_VERSION, build_messages
from .providers import StructuredMode
from .schemas import AnalysisResult, CaptureReport

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

    @property
    def ok(self) -> bool:
        return self.analysis is not None

    def guidance(self) -> str:
        """What the user should do about a failure."""
        return explain_failure(self.failure, self.config)


def analyze_capture(
    snapshot: FlowSnapshot,
    capture_path: str | Path,
    config: AIConfig | None = None,
    client: LLMClient | None = None,
) -> AnalysisOutcome:
    """Analyse one capture. Never raises for an expected failure.

    ``client`` may be injected — pass a
    :class:`~ai.llm_client.FakeLLMClient` to run the full pipeline offline.
    """
    cfg = config or AIConfig.from_env()
    started = time.monotonic()

    # --- deterministic stages: always run, never need a network ------------
    report = build_capture_report(snapshot, capture_path, cfg)

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
    messages = build_messages(report, schema)
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
            elapsed_seconds=time.monotonic() - started,
        )

    analysis = result.parsed

    # --- post-validation: the mechanical hallucination check ---------------
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
        elapsed_seconds=time.monotonic() - started,
    )
