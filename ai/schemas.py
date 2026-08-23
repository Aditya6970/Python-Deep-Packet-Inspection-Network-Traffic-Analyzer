"""Pydantic models for the AI analysis layer.

Two families of model live here:

**Input** (:class:`FlowRecord`, :class:`CaptureReport`) — what the DPI engine
produced, sanitized and serialized. These are built by :mod:`ai.extractor` and
are the *only* thing that reaches a prompt.

**Output** (:class:`AnalysisResult`, :class:`Indicator`, :class:`Action`) — the
shape the model is constrained to produce, and validated against on return.

Design notes
------------
Every field maps to something the engine actually measures. There is no
``duration`` field: ``Connection.first_seen``/``last_seen`` are
:func:`time.monotonic` values recording *processing* time, not capture time, so
a duration derived from them would be a fabricated number for the model to
reason from.

There is no ``detected_application`` in the output either. The engine already
determines that deterministically from real TLS SNI bytes; asking the model to
re-derive it would invite it to contradict ground truth. It receives the
application as *input*.

Observed facts, interpretation and uncertainty are **separate list fields**
rather than prose. Structure enforces the distinction better than an
instruction can: a claim placed under ``interpretation`` is self-labelling.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "SCHEMA_VERSION",
    "TransportProtocol",
    "TrafficType",
    "RiskLevel",
    "Severity",
    "Priority",
    "FlowRecord",
    "CaptureTotals",
    "CaptureReport",
    "Indicator",
    "Action",
    "AnalysisResult",
]

#: Bumped when the output shape changes, so stored results stay parseable.
SCHEMA_VERSION = "1.0"


# ===========================================================================
# Enumerations
# ===========================================================================
class TransportProtocol(str, Enum):
    """Transport protocol, as the engine classifies it."""

    TCP = "TCP"
    UDP = "UDP"
    OTHER = "OTHER"


class TrafficType(str, Enum):
    """Coarse category of what the capture appears to contain."""

    WEB_BROWSING = "web_browsing"
    STREAMING = "streaming"
    MESSAGING = "messaging"
    DNS = "dns"
    FILE_TRANSFER = "file_transfer"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Assessed risk.  ``UNKNOWN`` is a valid, expected answer."""

    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Severity of a single indicator."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Priority(str, Enum):
    """Priority of a recommended action."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ===========================================================================
# Input models -- what the DPI engine produced
# ===========================================================================
class FlowRecord(BaseModel):
    """One sanitized network flow.

    Built by :mod:`ai.extractor` from a :class:`dpi.types.Connection`. Only
    named fields are copied, so raw packet bytes have no structural path into
    this model even by accident.
    """

    model_config = ConfigDict(extra="forbid")

    flow_id: int = Field(
        ge=0,
        description="Stable index for this capture. Reference flows by this, not by address.",
    )
    protocol: TransportProtocol
    dst_port: int = Field(ge=0, le=65535)
    src_port: int = Field(ge=0, le=65535)

    server_name: str | None = Field(
        default=None,
        max_length=253,  # DNS maximum
        description="Hostname from TLS SNI, HTTP Host or a DNS query. Untrusted, sanitized.",
    )
    application: str = Field(description="Application identified by the DPI engine.")
    state: str = Field(description="Connection state: NEW, ESTABLISHED, CLASSIFIED, BLOCKED, CLOSED.")
    verdict: str = Field(description="Engine decision: FORWARD or DROP.")

    packets_out: int = Field(ge=0, description="Packets in the direction that opened the flow.")
    packets_in: int = Field(ge=0, description="Packets on the return path.")
    bytes_out: int = Field(ge=0)
    bytes_in: int = Field(ge=0)

    syn_seen: bool = False
    syn_ack_seen: bool = False
    fin_seen: bool = False

    src_ip: str | None = Field(default=None, description="Omitted or pseudonymised per redaction policy.")
    dst_ip: str | None = Field(default=None, description="Omitted or pseudonymised per redaction policy.")

    # NOTE: deliberately absent -- `duration`, and any payload bytes.

    @field_validator("server_name")
    @classmethod
    def _no_control_characters(cls, v: str | None) -> str | None:
        """Reject control characters outright.

        Sanitization proper happens in :mod:`ai.redaction`; this is a
        last-line assertion that nothing unsanitized reached the model.
        """
        if v is None:
            return None
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in v):
            raise ValueError("server_name contains control characters")
        return v


class CaptureTotals(BaseModel):
    """Capture-wide counters straight from the engine."""

    model_config = ConfigDict(extra="forbid")

    total_packets: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    tcp_packets: int = Field(ge=0)
    udp_packets: int = Field(ge=0)
    forwarded_packets: int = Field(ge=0)
    dropped_packets: int = Field(ge=0)
    total_flows: int = Field(ge=0)
    flows_included: int = Field(ge=0, description="Flows in this report; may be capped.")


class CaptureReport(BaseModel):
    """The complete sanitized picture sent for analysis."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    capture_name: str = Field(description="Source file name only -- never a full path.")
    totals: CaptureTotals
    application_distribution: dict[str, int] = Field(
        default_factory=dict, description="Application name -> flow count."
    )
    top_server_names: list[str] = Field(
        default_factory=list, description="Most frequently seen hostnames, sanitized."
    )
    blocking_rules_active: dict[str, int] = Field(
        default_factory=dict, description="Counts of configured IP/app/domain/port rules."
    )
    flows: list[FlowRecord] = Field(default_factory=list)
    redaction_mode: str = Field(description="Which IP policy produced this report.")
    notes: list[str] = Field(
        default_factory=list,
        description="Extraction caveats, e.g. that flows were capped.",
    )

    def flow_ids(self) -> set[int]:
        """The set of flow ids present, for validating model references."""
        return {f.flow_id for f in self.flows}


# ===========================================================================
# Output models -- what the model is constrained to produce
# ===========================================================================
class Indicator(BaseModel):
    """One noteworthy observation about the capture."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=500)
    severity: Severity
    supporting_flow_ids: list[int] = Field(
        default_factory=list,
        description="Flow ids from the input that evidence this. Must exist.",
    )
    is_inference: bool = Field(
        description="False only if this is a direct restatement of supplied data."
    )


class Action(BaseModel):
    """One recommended next step."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=500)
    priority: Priority
    rationale: str = Field(min_length=1, max_length=500)


class AnalysisResult(BaseModel):
    """Structured analysis of one capture.

    The three-way split between :attr:`observed_facts`,
    :attr:`interpretation` and :attr:`uncertainties` is the core of the design.
    It makes the model commit, per claim, to whether it is restating data or
    reasoning beyond it.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"

    summary: str = Field(
        min_length=1, max_length=2000, description="Two to four sentences, plain language."
    )

    observed_facts: list[str] = Field(
        default_factory=list,
        description="ONLY restatements of the supplied data. No inference.",
    )
    interpretation: list[str] = Field(
        default_factory=list,
        description="Inferences drawn from the data, explicitly labelled as such.",
    )
    uncertainties: list[str] = Field(
        default_factory=list,
        description="What this data cannot determine. An empty list is suspicious.",
    )

    traffic_type: TrafficType
    risk_level: RiskLevel
    risk_rationale: str = Field(min_length=1, max_length=1000)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]

    indicators: list[Indicator] = Field(default_factory=list)
    recommended_actions: list[Action] = Field(default_factory=list)
    notable_flow_ids: list[int] = Field(
        default_factory=list, description="Flow ids worth attention. Must exist in the input."
    )

    def validate_flow_references(self, known_ids: set[int]) -> list[str]:
        """Check every referenced flow id exists in the input.

        This is the mechanical hallucination check: if the model invents flow
        99 for a capture with 27 flows, it is caught here rather than trusted.
        Returns a list of human-readable problems; empty means clean.
        """
        problems: list[str] = []

        unknown = sorted(set(self.notable_flow_ids) - known_ids)
        if unknown:
            problems.append(f"notable_flow_ids references unknown flows: {unknown}")

        for i, ind in enumerate(self.indicators):
            bad = sorted(set(ind.supporting_flow_ids) - known_ids)
            if bad:
                problems.append(f"indicators[{i}] references unknown flows: {bad}")

        return problems
