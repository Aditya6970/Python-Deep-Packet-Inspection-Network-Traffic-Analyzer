"""Render an :class:`~ai.analyzer.AnalysisOutcome` as console text.

Deterministic and offline. Kept apart from :mod:`ai.analyzer` so presentation
can change without touching orchestration, and so rendering can be tested
without any LLM involvement.

The box-drawing frame matches the existing DPI reports. Console encoding is
handled by ``dpi.console``, which the ``dpi`` package applies on import.
"""

from __future__ import annotations

from .analyzer import AnalysisOutcome
from .schemas import AnalysisResult

__all__ = ["render_analysis", "render_failure", "render"]

_WIDTH = 62


def _rule(char: str = "═") -> str:
    return char * _WIDTH


def _section(title: str) -> list[str]:
    return [f"╠{_rule()}╣", f"║ {title:<{_WIDTH - 2}} ║", f"╠{_rule()}╣"]


def _wrap(text: str, indent: str = "  ", width: int = 76) -> list[str]:
    """Wrap without importing textwrap, keeping output predictable."""
    words = text.split()
    lines: list[str] = []
    current = indent
    for word in words:
        if len(current) + len(word) + 1 > width and current.strip():
            lines.append(current.rstrip())
            current = indent + word + " "
        else:
            current += word + " "
    if current.strip():
        lines.append(current.rstrip())
    return lines


def _bullets(items: list[str], marker: str = "-") -> list[str]:
    out: list[str] = []
    for item in items:
        wrapped = _wrap(item, indent="    ")
        if wrapped:
            first = wrapped[0].lstrip()
            out.append(f"  {marker} {first}")
            out.extend(wrapped[1:])
    return out or ["  (none)"]


def render_analysis(outcome: AnalysisOutcome) -> str:
    """Render a successful analysis."""
    a: AnalysisResult = outcome.analysis  # type: ignore[assignment]
    r = outcome.report

    lines: list[str] = []
    lines.append(f"╔{_rule()}╗")
    lines.append(f"║ {'AI CAPTURE ANALYSIS':^{_WIDTH - 2}} ║")
    lines.append(f"╠{_rule()}╣")
    lines.append(f"║ {('Capture:    ' + r.capture_name):<{_WIDTH - 2}} ║")
    lines.append(
        f"║ {('Provider:   ' + (outcome.provider or 'unknown')):<{_WIDTH - 2}} ║"
    )
    lines.append(f"║ {('Model:      ' + (outcome.model or 'unknown')):<{_WIDTH - 2}} ║")
    lines.append(
        f"║ {('Traffic:    ' + a.traffic_type.value):<{_WIDTH - 2}} ║"
    )
    lines.append(
        f"║ {('Risk:       ' + a.risk_level.value + f'   (confidence {a.confidence:.0%})'):<{_WIDTH - 2}} ║"
    )
    lines.append(f"║ {('Flows sent: ' + str(r.totals.flows_included) + ' of ' + str(r.totals.total_flows)):<{_WIDTH - 2}} ║")
    lines.append(f"╚{_rule()}╝")

    lines.append("")
    lines.append("SUMMARY")
    lines.extend(_wrap(a.summary))

    lines.append("")
    lines.append("OBSERVED FACTS  (restated from the capture data)")
    lines.extend(_bullets(a.observed_facts))

    lines.append("")
    lines.append("INTERPRETATION  (inference beyond the data)")
    lines.extend(_bullets(a.interpretation))

    lines.append("")
    lines.append("UNCERTAINTIES  (what this data cannot establish)")
    lines.extend(_bullets(a.uncertainties, marker="?"))

    if a.indicators:
        lines.append("")
        lines.append("INDICATORS")
        for ind in a.indicators:
            tag = "inferred" if ind.is_inference else "observed"
            flows = (
                f"  [flows {', '.join(str(i) for i in ind.supporting_flow_ids)}]"
                if ind.supporting_flow_ids
                else ""
            )
            head = f"  [{ind.severity.value:<6}] ({tag}){flows}"
            lines.append(head)
            lines.extend(_wrap(ind.description, indent="      "))

    lines.append("")
    lines.append("RISK RATIONALE")
    lines.extend(_wrap(a.risk_rationale))

    if a.recommended_actions:
        lines.append("")
        lines.append("RECOMMENDED ACTIONS")
        for act in a.recommended_actions:
            lines.append(f"  [{act.priority.value:<6}] {act.description}")
            lines.extend(_wrap(f"why: {act.rationale}", indent="      "))

    if a.notable_flow_ids:
        lines.append("")
        lines.append(
            "NOTABLE FLOWS: " + ", ".join(str(i) for i in a.notable_flow_ids)
        )

    if outcome.warnings:
        lines.append("")
        lines.append("VALIDATION WARNINGS")
        lines.extend(_bullets(outcome.warnings, marker="!"))

    if r.notes:
        lines.append("")
        lines.append("EXTRACTION NOTES")
        lines.extend(_bullets(r.notes))

    lines.append("")
    lines.append(
        f"[provider {outcome.provider or '?'} | prompt v{outcome.prompt_version} | "
        f"schema v{a.schema_version} | {outcome.attempts} attempt(s) | "
        f"{outcome.elapsed_seconds:.1f}s | ip mode: {r.redaction_mode}]"
    )

    return "\n".join(lines) + "\n"


def render_failure(outcome: AnalysisOutcome) -> str:
    """Render a skipped or failed analysis, without alarm.

    A failure here is not an error condition for the program: the DPI results
    printed above remain complete and correct.
    """
    reason = outcome.failure.value if outcome.failure else "unknown"
    lines: list[str] = []
    provider_line = (
        f"  Provider: {outcome.provider}" + (f" ({outcome.model})" if outcome.model else "")
        if outcome.provider
        else ""
    )
    lines.append(f"╔{_rule()}╗")
    lines.append(f"║ {'AI CAPTURE ANALYSIS - SKIPPED':^{_WIDTH - 2}} ║")
    lines.append(f"╚{_rule()}╝")
    lines.append("")
    if provider_line:
        lines.append(provider_line)
    lines.append(f"  Reason: {reason}")
    if outcome.detail and outcome.detail != outcome.guidance():
        lines.extend(_wrap(outcome.detail))
    lines.append("")
    lines.extend(_wrap(outcome.guidance()))
    lines.append("")
    lines.append("  The DPI analysis above is complete and unaffected.")
    return "\n".join(lines) + "\n"


def render(outcome: AnalysisOutcome) -> str:
    """Render whichever outcome occurred."""
    return render_analysis(outcome) if outcome.ok else render_failure(outcome)
