"""Deterministic extraction of retrieval signals from a sanitized capture report.

What this does
--------------
Reads an :class:`~ai.schemas.CaptureReport` -- the redacted, validated view the
AI layer already builds -- and emits a small set of structured **observations**
about what the DPI engine measured.  Those observations are what a later step
will turn into retrieval queries.

What this does **not** do: build queries, embed anything, search a vector
store, call an LLM, or reach the network.  It also does not reach *backwards*:
nothing here imports :mod:`dpi`, so the dependency direction is::

    dpi/  ->  ai/extractor.py  ->  CaptureReport  ->  ai/rag/signals.py

Reading the report rather than the raw :class:`~dpi.dpi_engine.FlowSnapshot` is
a privacy decision, not a convenience.  The snapshot holds real addresses and
live ``Connection`` objects; the report is what survives
:mod:`ai.redaction` -- hostnames sanitized, addresses omitted or pseudonymised
per ``redaction_mode``, payload bytes structurally absent.  Extracting signals
from the snapshot would quietly route around a boundary the project already
built and tests.

Signals are observations, not verdicts
--------------------------------------
``unknown_app_share`` means "the classifier had no SNI to work with", not "this
traffic is malicious".  ``scan_port_fanout`` means "one source reached many
ports", not "someone is scanning you".  Every :class:`Signal` therefore carries
a mandatory :attr:`Signal.does_not_prove` field, and severity ranks *how much
the observation is worth looking into*, never how dangerous it is.  The
knowledge corpus supplies context and the LLM supplies interpretation; this
layer only reports what the numbers say.

The vocabulary is fixed elsewhere, on purpose
---------------------------------------------
:class:`SignalType` mirrors ``KNOWN_SIGNALS`` in :mod:`ai.rag.documents`, and a
module-level assertion keeps the two identical.  Step 1 pinned those names so a
knowledge document could not claim to answer a signal that will never exist;
this is the other half of that contract.  Adding a signal means adding it in
both places, in one commit.

What cannot be computed, and is therefore absent
------------------------------------------------
:class:`~ai.schemas.FlowRecord` carries no timestamps, no durations and no
payload bytes -- deliberately, because ``Connection.first_seen``/``last_seen``
measure *processing* time rather than capture time.  So there is no beaconing
signal, no periodicity signal, no rate or jitter signal and no protocol
fingerprint signal.  Those are not oversights and must not be added by
inferring a clock that does not exist.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Final, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..schemas import CaptureReport, FlowRecord, Severity, TransportProtocol
from .documents import KNOWN_SIGNALS

__all__ = [
    "COMMON_PORTS",
    "SIGNAL_SCHEMA_VERSION",
    "CaptureProfile",
    "Signal",
    "SignalConfig",
    "SignalReport",
    "SignalType",
    "SourceField",
    "extract_signals",
    "shannon_entropy",
]

#: Bumped when the signal report shape changes.
SIGNAL_SCHEMA_VERSION: Final[str] = "1.0"


# ===========================================================================
# Vocabulary
# ===========================================================================
class SignalType(str, Enum):
    """The controlled signal vocabulary.

    Identical to ``ai.rag.documents.KNOWN_SIGNALS`` -- see the assertion below.
    """

    # DNS behaviour
    DNS_HIGH_VOLUME = "dns_high_volume"
    DNS_HIGH_CARDINALITY = "dns_high_cardinality"
    DNS_ANOMALOUS_LABEL = "dns_anomalous_label"
    # Connection attempts
    SCAN_PORT_FANOUT = "scan_port_fanout"
    SCAN_HALF_OPEN = "scan_half_open"
    # Classification outcomes
    UNKNOWN_APP_SHARE = "unknown_app_share"
    TLS_WITHOUT_SNI = "tls_without_sni"
    PLAINTEXT_HTTP = "plaintext_http"
    QUIC_PRESENT = "quic_present"
    # Volume and destination shape
    UPLOAD_ASYMMETRY = "upload_asymmetry"
    NONSTANDARD_PORT_EGRESS = "nonstandard_port_egress"
    # Engine decisions
    BLOCKED_TRAFFIC_PRESENT = "blocked_traffic_present"
    # Baseline
    BASELINE_WEB_BROWSING = "baseline_web_browsing"


# The corpus and the extractor must never drift apart.  A knowledge document
# declaring `applies_to: [dns_beaconing]` was rejected in step 1; a signal type
# with no document to retrieve would be the mirror-image mistake.
assert {t.value for t in SignalType} == set(KNOWN_SIGNALS), (
    "SignalType and ai.rag.documents.KNOWN_SIGNALS have diverged: "
    f"{sorted({t.value for t in SignalType} ^ set(KNOWN_SIGNALS))}"
)


class SourceField(str, Enum):
    """Report fields a signal is allowed to cite as its cause.

    A closed set, so "which DPI data produced this?" is answerable mechanically
    rather than by reading prose.  Every member names a field that genuinely
    exists on :class:`~ai.schemas.CaptureReport`,
    :class:`~ai.schemas.CaptureTotals` or :class:`~ai.schemas.FlowRecord`;
    anything else is rejected by the model.
    """

    # CaptureReport
    APPLICATION_DISTRIBUTION = "report.application_distribution"
    BLOCKING_RULES_ACTIVE = "report.blocking_rules_active"
    REDACTION_MODE = "report.redaction_mode"
    TOP_SERVER_NAMES = "report.top_server_names"
    # CaptureTotals
    TOTAL_FLOWS = "report.totals.total_flows"
    FLOWS_INCLUDED = "report.totals.flows_included"
    DROPPED_PACKETS = "report.totals.dropped_packets"
    TCP_PACKETS = "report.totals.tcp_packets"
    UDP_PACKETS = "report.totals.udp_packets"
    # FlowRecord
    FLOW_ID = "flows.flow_id"
    PROTOCOL = "flows.protocol"
    DST_PORT = "flows.dst_port"
    SRC_PORT = "flows.src_port"
    SERVER_NAME = "flows.server_name"
    APPLICATION = "flows.application"
    STATE = "flows.state"
    VERDICT = "flows.verdict"
    PACKETS_OUT = "flows.packets_out"
    PACKETS_IN = "flows.packets_in"
    BYTES_OUT = "flows.bytes_out"
    BYTES_IN = "flows.bytes_in"
    SYN_SEEN = "flows.syn_seen"
    SYN_ACK_SEEN = "flows.syn_ack_seen"
    FIN_SEEN = "flows.fin_seen"
    SRC_IP = "flows.src_ip"
    DST_IP = "flows.dst_ip"


#: Ports whose purpose is documented and unremarkable, used by
#: ``nonstandard_port_egress`` as its "expected" set.  Not a safety judgement:
#: a destination being on this list says only that the port is conventional.
COMMON_PORTS: Final[frozenset[int]] = frozenset({
    20, 21, 22, 25, 53, 67, 68, 80, 110, 123, 143, 161, 389, 443, 445,
    465, 587, 636, 853, 993, 995, 1900, 3478, 5353, 8080, 8443,
})

#: The classifier's name for a flow it could not identify.  Matches
#: ``dpi.types.app_type_to_string(AppType.UNKNOWN)``.
_UNKNOWN_APPLICATION: Final[str] = "Unknown"

_DNS_PORT: Final[int] = 53
_HTTP_PORT: Final[int] = 80
_TLS_PORT: Final[int] = 443

_EVIDENCE_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")
_SIGNAL_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z_]+#[0-9a-f]{12}$")
_MAX_EVIDENCE_STRING: Final[int] = 200
_MAX_EVIDENCE_LIST: Final[int] = 64


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass(frozen=True, slots=True)
class SignalConfig:
    """Detection thresholds.

    Every threshold is named, defaulted, documented and overridable.  They are
    starting points chosen to be defensible, not values tuned against a
    labelled dataset -- which is exactly what the corpus's
    ``suspicious-dns-indicators`` document says about them, and why crossing
    one raises *relevance*, never a verdict.

    Absolute minimums sit beside every ratio on purpose.  A ratio alone lies on
    small captures: one DNS flow out of three is a 33% share and means nothing.
    """

    #: Unknown-application share: both the ratio and a floor on the count.
    unknown_app_min_share: float = 0.20
    unknown_app_min_flows: int = 3

    #: DNS volume: DNS flows as a share of all flows, with a floor.
    dns_min_share: float = 0.15
    dns_min_flows: int = 5

    #: DNS name cardinality: distinct names / DNS flows.  Browsing reuses
    #: names because of caching; encoded data cannot.
    dns_cardinality_min_ratio: float = 0.90
    dns_cardinality_min_flows: int = 5

    #: DNS label shape.  63 octets is the protocol maximum, so a label past 30
    #: is already unusual; 3.5 bits/char is well above English text.
    dns_label_min_length: int = 30
    dns_label_min_entropy: float = 3.5

    #: Connection fan-out: distinct destination ports reached by one source.
    scan_min_distinct_ports: int = 8

    #: Half-open connections: SYN sent, no SYN-ACK seen.
    half_open_min_flows: int = 3
    half_open_min_ratio: float = 0.50

    #: Upload asymmetry: bytes_out / bytes_in, with a byte floor so a tiny
    #: flow cannot trigger it.
    upload_min_ratio: float = 3.0
    upload_min_bytes_out: int = 4096

    #: Non-standard destination ports carrying real traffic.
    nonstandard_min_flows: int = 2
    nonstandard_min_bytes: int = 512

    #: Minimum flows before the plain-presence signals fire.
    presence_min_flows: int = 1

    def as_dict(self) -> dict[str, float | int]:
        """Threshold values, sorted -- recorded in every report for audit."""
        return {
            name: getattr(self, name)
            for name in sorted(self.__dataclass_fields__)  # type: ignore[attr-defined]
        }


# ===========================================================================
# Models
# ===========================================================================
EvidenceValue = int | float | str | bool | list[int] | list[str]


class Signal(BaseModel):
    """One observation about a capture.

    Immutable, strictly validated, and required to say what it does *not*
    establish.  That last field is not documentation politeness: a signal
    without it reads as a finding, and a list of findings reads as a verdict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    signal_id: str = Field(
        min_length=3,
        description="Deterministic: <signal_type>#<12 hex of an evidence hash>.",
    )
    signal_type: SignalType
    severity: Severity = Field(
        description="How much this is worth looking into. Never a claim of maliciousness."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How firmly the data supports the observation, not how bad it is.",
    )
    summary: str = Field(
        min_length=10, max_length=400,
        description="What was observed, in plain language, without a verdict.",
    )
    does_not_prove: str = Field(
        min_length=10, max_length=400,
        description="The conclusion a reader must not draw from this signal alone.",
    )
    evidence: dict[str, EvidenceValue] = Field(
        min_length=1, description="Machine-readable numbers behind the observation."
    )
    flow_ids: tuple[int, ...] = Field(
        default=(),
        description="Flows evidencing this, sorted ascending. Empty for capture-wide signals.",
    )
    source_fields: tuple[SourceField, ...] = Field(
        min_length=1, description="The report fields this signal was computed from."
    )

    # -- validators ---------------------------------------------------------
    @field_validator("signal_id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not _SIGNAL_ID_PATTERN.match(v):
            raise ValueError(
                f"signal_id {v!r} is malformed; expected <signal_type>#<12 hex digits>"
            )
        return v

    @field_validator("evidence", mode="before")
    @classmethod
    def _no_raw_bytes(cls, v: object) -> object:
        """Reject bytes *before* pydantic can coerce them into a string.

        This runs in ``before`` mode on purpose: in lax mode pydantic will
        happily decode ``b"\\x16\\x03\\x01"`` into the ``str`` branch of the
        evidence union, so a payload fragment would slip through a validator
        that only inspects the coerced value.  Packet bytes must never reach
        an evidence field, so the check has to happen first.
        """
        if isinstance(v, dict):
            for key, value in v.items():
                if isinstance(value, (bytes, bytearray, memoryview)):
                    raise ValueError(
                        f"evidence[{key!r}] is raw bytes; packet data must never appear here"
                    )
                if isinstance(value, (list, tuple)) and any(
                    isinstance(item, (bytes, bytearray, memoryview)) for item in value
                ):
                    raise ValueError(
                        f"evidence[{key!r}] contains raw bytes; packet data must never "
                        "appear here"
                    )
        return v

    @field_validator("evidence")
    @classmethod
    def _evidence_is_clean(cls, v: dict[str, EvidenceValue]) -> dict[str, EvidenceValue]:
        """Keep evidence small, machine-readable and free of anything sensitive."""
        if not v:
            raise ValueError("evidence must not be empty")
        for key, value in v.items():
            if not _EVIDENCE_KEY_PATTERN.match(key):
                raise ValueError(
                    f"evidence key {key!r} must be lowercase snake_case"
                )
            if isinstance(value, (bytes, bytearray, memoryview)):
                raise ValueError(
                    f"evidence[{key!r}] is raw bytes; packet data must never appear here"
                )
            if isinstance(value, str) and len(value) > _MAX_EVIDENCE_STRING:
                raise ValueError(
                    f"evidence[{key!r}] is {len(value)} characters; evidence must stay concise"
                )
            if isinstance(value, list):
                if len(value) > _MAX_EVIDENCE_LIST:
                    raise ValueError(
                        f"evidence[{key!r}] has {len(value)} items; cap is {_MAX_EVIDENCE_LIST}"
                    )
                for item in value:
                    if isinstance(item, str) and len(item) > _MAX_EVIDENCE_STRING:
                        raise ValueError(f"evidence[{key!r}] contains an over-long string")
        return v

    @field_validator("flow_ids")
    @classmethod
    def _sorted_unique(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(flow_id < 0 for flow_id in v):
            raise ValueError("flow_ids must be non-negative")
        if len(set(v)) != len(v):
            raise ValueError("flow_ids contains duplicates")
        if list(v) != sorted(v):
            raise ValueError("flow_ids must be sorted ascending for determinism")
        return v

    @field_validator("source_fields")
    @classmethod
    def _sorted_unique_fields(cls, v: tuple[SourceField, ...]) -> tuple[SourceField, ...]:
        values = [field.value for field in v]
        if len(set(values)) != len(values):
            raise ValueError("source_fields contains duplicates")
        if values != sorted(values):
            raise ValueError("source_fields must be sorted for determinism")
        return v

    @model_validator(mode="after")
    def _id_matches_type(self) -> Signal:
        if not self.signal_id.startswith(f"{self.signal_type.value}#"):
            raise ValueError(
                f"signal_id {self.signal_id!r} does not begin with its signal_type"
            )
        return self

    # -- helpers ------------------------------------------------------------
    def validate_flow_ids(self, known: frozenset[int]) -> list[str]:
        """Report flow ids that do not exist in the capture.

        The mechanical check that a signal cannot cite traffic that was never
        observed -- the same guarantee
        :meth:`~ai.schemas.AnalysisResult.validate_flow_references` gives on
        the output side.
        """
        unknown = sorted(set(self.flow_ids) - known)
        return [] if not unknown else [
            f"{self.signal_type.value} references flows not in the capture: {unknown}"
        ]


class CaptureProfile(BaseModel):
    """Deterministic distributions describing the capture as a whole.

    Context rather than signal: a distribution is never "detected", it simply
    is.  Keeping it out of :class:`Signal` avoids inventing a signal type whose
    detection logic would be "always true".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_distribution: dict[str, int] = Field(default_factory=dict)
    protocol_distribution: dict[str, int] = Field(default_factory=dict)
    verdict_distribution: dict[str, int] = Field(default_factory=dict)
    state_distribution: dict[str, int] = Field(default_factory=dict)
    top_destination_ports: list[tuple[int, int]] = Field(
        default_factory=list, description="(port, flow count), most frequent first."
    )


class SignalReport(BaseModel):
    """Every signal found in one capture, plus the context to reproduce it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = SIGNAL_SCHEMA_VERSION
    generated_from: str = Field(
        description="The input this was derived from, e.g. 'ai.schemas.CaptureReport v1.0'."
    )
    capture_name: str
    redaction_mode: str = Field(
        description="Copied from the report, so a reader knows what addresses mean here."
    )

    flow_count: int = Field(ge=0, description="Flows present in the source report.")
    total_flow_count: int = Field(ge=0, description="Flows the engine saw, before capping.")
    signal_count: int = Field(ge=0)
    signals: tuple[Signal, ...] = ()
    profile: CaptureProfile = Field(default_factory=CaptureProfile)
    thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="The SignalConfig values used, so a report can be reproduced.",
    )

    # NOTE: there is deliberately no `generated_at`.  A timestamp would make
    # two runs over identical input differ, which would defeat every
    # determinism guarantee this layer offers.

    @model_validator(mode="after")
    def _consistent(self) -> SignalReport:
        if self.signal_count != len(self.signals):
            raise ValueError("signal_count does not match the number of signals")

        types = [signal.signal_type for signal in self.signals]
        if len(set(types)) != len(types):
            raise ValueError("a signal type appears more than once in one report")

        keys = [(-_SEVERITY_RANK[s.severity], -round(s.confidence, 6), s.signal_type.value)
                for s in self.signals]
        if keys != sorted(keys):
            raise ValueError(
                "signals are not in the documented order "
                "(severity desc, confidence desc, signal_type asc)"
            )
        return self

    def validate_flow_ids(self, known: frozenset[int]) -> list[str]:
        """Every problem found across every signal.  Empty means clean."""
        problems: list[str] = []
        for signal in self.signals:
            problems.extend(signal.validate_flow_ids(known))
        return problems

    def by_type(self, signal_type: SignalType) -> Signal | None:
        return next((s for s in self.signals if s.signal_type is signal_type), None)

    def types(self) -> tuple[str, ...]:
        return tuple(signal.signal_type.value for signal in self.signals)

    def to_json(self) -> str:
        """Stable JSON: sorted keys, fixed separators, no clock anywhere."""
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True,
                          ensure_ascii=False)


#: Severity ordering used for ranking signals.  Higher means "look at this
#: first", never "this is more dangerous".
_SEVERITY_RANK: Final[dict[Severity, int]] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
}


# ===========================================================================
# Helpers
# ===========================================================================
def shannon_entropy(text: str) -> float:
    """Shannon entropy of a string in bits per character.

    English words land near 3.0; base32 or hex encoding of random bytes lands
    well above 4.0.  Used only on the sanitized ``server_name``, never on
    payload.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def _leftmost_label(hostname: str) -> str:
    return hostname.split(".", 1)[0] if hostname else ""


def _registrable_parent(hostname: str) -> str:
    """The last two labels of a hostname.

    A deliberate approximation: a real public-suffix list would be a dependency
    and a data file to keep current, and the grouping only needs to be stable
    and explainable.  ``a.b.example.co.uk`` groups as ``co.uk``, which is wrong
    for that suffix and is stated here rather than hidden.
    """
    parts = [part for part in hostname.split(".") if part]
    return ".".join(parts[-2:]) if len(parts) >= 2 else hostname


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 6)


def _confidence(value: float, threshold: float, ceiling: float) -> float:
    """Map a measurement onto [0.5, 0.95] between its threshold and a ceiling.

    Deterministic and deliberately modest: a signal that only just crosses its
    threshold starts at 0.5, and no signal ever reaches certainty, because a
    threshold being crossed is evidence rather than proof.
    """
    if ceiling <= threshold:
        return 0.95 if value >= threshold else 0.5
    span = (value - threshold) / (ceiling - threshold)
    return round(min(0.95, max(0.5, 0.5 + 0.45 * span)), 4)


def _signal_id(signal_type: SignalType, evidence: dict[str, EvidenceValue]) -> str:
    """Deterministic id: the type plus a hash of its evidence.

    SHA-256 rather than :func:`hash`, whose value is salted per process; the
    same capture must always produce the same ids.  Including the evidence
    means an id changes when the underlying numbers change, which makes two
    reports diffable.
    """
    payload = json.dumps(evidence, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(f"{signal_type.value}\n{payload}".encode("utf-8")).hexdigest()
    return f"{signal_type.value}#{digest[:12]}"


def _make(
    signal_type: SignalType,
    severity: Severity,
    confidence: float,
    summary: str,
    does_not_prove: str,
    evidence: dict[str, EvidenceValue],
    source_fields: Sequence[SourceField],
    flow_ids: Sequence[int] = (),
) -> Signal:
    """Assemble a signal with the ordering and id rules applied consistently."""
    ordered_evidence = {key: evidence[key] for key in sorted(evidence)}
    return Signal(
        signal_id=_signal_id(signal_type, ordered_evidence),
        signal_type=signal_type,
        severity=severity,
        confidence=confidence,
        summary=summary,
        does_not_prove=does_not_prove,
        evidence=ordered_evidence,
        flow_ids=tuple(sorted(set(flow_ids))),
        source_fields=tuple(sorted(set(source_fields), key=lambda f: f.value)),
    )


def _is_dns(flow: FlowRecord) -> bool:
    return flow.protocol is TransportProtocol.UDP and flow.dst_port == _DNS_PORT


def _capped(flow_ids: Sequence[int]) -> list[int]:
    """Trim a flow-id list for evidence, keeping the smallest ids.

    The full list still lives on ``Signal.flow_ids``; evidence stays readable.
    """
    return sorted(flow_ids)[:_MAX_EVIDENCE_LIST]


# ===========================================================================
# Detectors
# ===========================================================================
# Each detector is a pure function of (report, config) returning one Signal or
# None.  They never look at anything but the report, never mutate it, and are
# listed in a fixed order so the extractor's behaviour is easy to follow.
Detector = Callable[[CaptureReport, SignalConfig], "Signal | None"]


def _detect_unknown_app_share(report: CaptureReport, cfg: SignalConfig) -> Signal | None:
    """Flows the classifier could not name.

    Source: ``flows.application`` (and ``report.application_distribution`` for
    the capture-wide count).  The engine names a flow from TLS SNI, an HTTP
    Host header or a DNS query name; with none of those present the flow is
    ``Unknown`` however ordinary it is.
    """
    unknown = [f for f in report.flows if f.application == _UNKNOWN_APPLICATION]
    total = len(report.flows)
    share = _ratio(len(unknown), total)

    if len(unknown) < cfg.unknown_app_min_flows or share < cfg.unknown_app_min_share:
        return None

    severity = Severity.MEDIUM if share >= 0.5 else Severity.LOW
    return _make(
        SignalType.UNKNOWN_APP_SHARE,
        severity,
        _confidence(share, cfg.unknown_app_min_share, 0.8),
        f"{len(unknown)} of {total} flows ({share:.0%}) were not classified by the engine.",
        "Unclassified is a classifier outcome, not a risk finding. Encrypted Client "
        "Hello, QUIC and any protocol the engine does not parse all land here.",
        {
            "unknown_flow_count": len(unknown),
            "total_flow_count": total,
            "ratio": share,
            "distribution_unknown_count":
                int(report.application_distribution.get(_UNKNOWN_APPLICATION, 0)),
        },
        (SourceField.APPLICATION, SourceField.APPLICATION_DISTRIBUTION,
         SourceField.FLOW_ID),
        [f.flow_id for f in unknown],
    )


def _detect_dns_high_volume(report: CaptureReport, cfg: SignalConfig) -> Signal | None:
    """DNS flows as a share of the capture.

    Source: ``flows.protocol`` and ``flows.dst_port`` (UDP/53).
    """
    dns = [f for f in report.flows if _is_dns(f)]
    total = len(report.flows)
    share = _ratio(len(dns), total)

    if len(dns) < cfg.dns_min_flows or share < cfg.dns_min_share:
        return None

    return _make(
        SignalType.DNS_HIGH_VOLUME,
        Severity.LOW,
        _confidence(share, cfg.dns_min_share, 0.6),
        f"{len(dns)} of {total} flows ({share:.0%}) are DNS queries over UDP port 53.",
        "High DNS volume is normal for a busy host, a capture taken at boot, or a "
        "capture taken on a resolver. On its own it indicates nothing.",
        {
            "dns_flow_count": len(dns),
            "total_flow_count": total,
            "ratio": share,
        },
        (SourceField.DST_PORT, SourceField.FLOW_ID, SourceField.PROTOCOL),
        [f.flow_id for f in dns],
    )


def _detect_dns_high_cardinality(report: CaptureReport, cfg: SignalConfig) -> Signal | None:
    """Distinct query names per DNS flow, and the busiest parent domain.

    Source: ``flows.server_name`` on DNS flows.  Caching means ordinary
    browsing reuses names; data encoded into names cannot repeat.
    """
    dns = [f for f in report.flows if _is_dns(f) and f.server_name]
    if len(dns) < cfg.dns_cardinality_min_flows:
        return None

    names = [f.server_name or "" for f in dns]
    distinct = len(set(names))
    ratio = _ratio(distinct, len(dns))
    if ratio < cfg.dns_cardinality_min_ratio:
        return None

    parents: Counter[str] = Counter(_registrable_parent(name) for name in names)
    top_parent, top_count = parents.most_common(1)[0]

    return _make(
        SignalType.DNS_HIGH_CARDINALITY,
        Severity.MEDIUM,
        _confidence(ratio, cfg.dns_cardinality_min_ratio, 1.0),
        f"{distinct} distinct query names across {len(dns)} DNS flows "
        f"({ratio:.0%} unique); the most frequent parent domain covers {top_count}.",
        "Security vendors, DNSBLs and some CDNs generate unique names by design, so "
        "high cardinality alone does not indicate encoding or exfiltration.",
        {
            "dns_flow_count": len(dns),
            "distinct_name_count": distinct,
            "ratio": ratio,
            "top_parent_domain": top_parent,
            "top_parent_name_count": top_count,
        },
        (SourceField.DST_PORT, SourceField.FLOW_ID, SourceField.PROTOCOL,
         SourceField.SERVER_NAME),
        [f.flow_id for f in dns],
    )


def _detect_dns_anomalous_label(report: CaptureReport, cfg: SignalConfig) -> Signal | None:
    """Long or high-entropy leftmost labels in DNS query names.

    Source: ``flows.server_name`` on DNS flows.  The name is already sanitized
    by :mod:`ai.redaction`; nothing here re-reads raw bytes.
    """
    flagged: list[tuple[FlowRecord, str, float]] = []
    for flow in report.flows:
        if not (_is_dns(flow) and flow.server_name):
            continue
        label = _leftmost_label(flow.server_name)
        entropy = shannon_entropy(label)
        if len(label) >= cfg.dns_label_min_length or entropy >= cfg.dns_label_min_entropy:
            flagged.append((flow, label, entropy))

    if not flagged:
        return None

    longest = max(len(label) for _, label, _ in flagged)
    highest = max(entropy for _, _, entropy in flagged)

    return _make(
        SignalType.DNS_ANOMALOUS_LABEL,
        Severity.MEDIUM,
        _confidence(highest, cfg.dns_label_min_entropy, 4.5),
        f"{len(flagged)} DNS query name(s) have an unusual leftmost label "
        f"(longest {longest} characters, highest entropy {highest:.2f} bits/char).",
        "Cloud providers and CDN edge nodes use machine-generated hostnames that look "
        "the same. Looking encoded is not the same as being encoded.",
        {
            "flagged_flow_count": len(flagged),
            "longest_label_length": longest,
            "highest_label_entropy": round(highest, 4),
            "length_threshold": cfg.dns_label_min_length,
            "entropy_threshold": cfg.dns_label_min_entropy,
        },
        (SourceField.DST_PORT, SourceField.FLOW_ID, SourceField.PROTOCOL,
         SourceField.SERVER_NAME),
        [flow.flow_id for flow, _, _ in flagged],
    )


def _detect_scan_port_fanout(report: CaptureReport, cfg: SignalConfig) -> Signal | None:
    """One source reaching many distinct destination ports.

    Source: ``flows.src_ip`` (pseudonymised or omitted per
    ``report.redaction_mode``) and ``flows.dst_port``.

    When ``src_ip`` is withheld entirely, every flow is grouped together and
    the evidence says so -- the fan-out is then a property of the capture
    rather than of a host, which is weaker but still worth surfacing.  Bypassing
    redaction to recover the real address is never an option.
    """
    grouped: defaultdict[str, list[FlowRecord]] = defaultdict(list)
    have_addresses = any(flow.src_ip for flow in report.flows)
    for flow in report.flows:
        grouped[flow.src_ip or "<withheld>"].append(flow)

    best_key, best_flows, best_ports = "", [], 0
    for key, flows in sorted(grouped.items()):
        ports = len({flow.dst_port for flow in flows})
        if ports > best_ports:
            best_key, best_flows, best_ports = key, flows, ports

    if best_ports < cfg.scan_min_distinct_ports:
        return None

    return _make(
        SignalType.SCAN_PORT_FANOUT,
        Severity.MEDIUM,
        _confidence(best_ports, cfg.scan_min_distinct_ports,
                    cfg.scan_min_distinct_ports * 4),
        f"One source reached {best_ports} distinct destination ports across "
        f"{len(best_flows)} flows.",
        "Port fan-out is produced by authorised vulnerability scanners, service "
        "discovery and ordinary multi-service clients as readily as by reconnaissance.",
        {
            "distinct_destination_ports": best_ports,
            "flow_count": len(best_flows),
            "grouped_by": "src_ip" if have_addresses else "capture",
            "redaction_mode": report.redaction_mode,
        },
        (SourceField.DST_PORT, SourceField.FLOW_ID, SourceField.REDACTION_MODE,
         SourceField.SRC_IP),
        [flow.flow_id for flow in best_flows],
    )


def _detect_scan_half_open(report: CaptureReport, cfg: SignalConfig) -> Signal | None:
    """TCP connections that were attempted but never answered.

    Source: ``flows.syn_seen``, ``flows.syn_ack_seen``, ``flows.protocol``.
    """
    tcp = [f for f in report.flows if f.protocol is TransportProtocol.TCP and f.syn_seen]
    if not tcp:
        return None

    half_open = [f for f in tcp if not f.syn_ack_seen]
    ratio = _ratio(len(half_open), len(tcp))

    if len(half_open) < cfg.half_open_min_flows or ratio < cfg.half_open_min_ratio:
        return None

    return _make(
        SignalType.SCAN_HALF_OPEN,
        Severity.MEDIUM,
        _confidence(ratio, cfg.half_open_min_ratio, 1.0),
        f"{len(half_open)} of {len(tcp)} TCP flows ({ratio:.0%}) sent a SYN that was "
        "never answered with a SYN-ACK.",
        "A closed port, a firewall drop or a destination that has gone away produces "
        "exactly this evidence. It is not by itself an attack.",
        {
            "half_open_count": len(half_open),
            "tcp_flow_count": len(tcp),
            "ratio": ratio,
        },
        (SourceField.FLOW_ID, SourceField.PROTOCOL, SourceField.SYN_ACK_SEEN,
         SourceField.SYN_SEEN),
        [f.flow_id for f in half_open],
    )


def _detect_tls_without_sni(report: CaptureReport, cfg: SignalConfig) -> Signal | None:
    """TCP 443 flows that carried no readable server name.

    Source: ``flows.protocol``, ``flows.dst_port``, ``flows.server_name``.
    """
    flows = [f for f in report.flows
             if f.protocol is TransportProtocol.TCP
             and f.dst_port == _TLS_PORT
             and not f.server_name]
    if len(flows) < cfg.presence_min_flows:
        return None

    tls_total = sum(1 for f in report.flows
                    if f.protocol is TransportProtocol.TCP and f.dst_port == _TLS_PORT)
    return _make(
        SignalType.TLS_WITHOUT_SNI,
        Severity.LOW,
        _confidence(_ratio(len(flows), tls_total), 0.0, 1.0),
        f"{len(flows)} of {tls_total} TLS flows on port 443 carried no readable "
        "server name.",
        "Encrypted Client Hello, a capture that began mid-session, or a resumed "
        "session all remove the SNI. The absence is about visibility, not intent.",
        {
            "tls_without_sni_count": len(flows),
            "tls_flow_count": tls_total,
            "ratio": _ratio(len(flows), tls_total),
        },
        (SourceField.DST_PORT, SourceField.FLOW_ID, SourceField.PROTOCOL,
         SourceField.SERVER_NAME),
        [f.flow_id for f in flows],
    )


def _detect_plaintext_http(report: CaptureReport, cfg: SignalConfig) -> Signal | None:
    """Traffic to TCP port 80.

    Source: ``flows.protocol``, ``flows.dst_port``.
    """
    flows = [f for f in report.flows
             if f.protocol is TransportProtocol.TCP and f.dst_port == _HTTP_PORT]
    if len(flows) < cfg.presence_min_flows:
        return None

    named = sorted({f.server_name for f in flows if f.server_name})
    return _make(
        SignalType.PLAINTEXT_HTTP,
        Severity.LOW,
        0.9,
        f"{len(flows)} flow(s) used unencrypted HTTP on TCP port 80.",
        "Plain HTTP is a confidentiality observation, not evidence of compromise. "
        "Captive portals, redirects to HTTPS and update checks all use it.",
        {
            "http_flow_count": len(flows),
            "total_flow_count": len(report.flows),
            "distinct_hostnames": len(named),
        },
        (SourceField.DST_PORT, SourceField.FLOW_ID, SourceField.PROTOCOL,
         SourceField.SERVER_NAME),
        [f.flow_id for f in flows],
    )


def _detect_quic_present(report: CaptureReport, cfg: SignalConfig) -> Signal | None:
    """UDP 443 traffic, which is almost always QUIC / HTTP-3.

    Source: ``flows.protocol``, ``flows.dst_port``.
    """
    flows = [f for f in report.flows
             if f.protocol is TransportProtocol.UDP and f.dst_port == _TLS_PORT]
    if len(flows) < cfg.presence_min_flows:
        return None

    return _make(
        SignalType.QUIC_PRESENT,
        Severity.INFO,
        0.9,
        f"{len(flows)} flow(s) used UDP port 443, consistent with QUIC / HTTP-3.",
        "QUIC carries a large share of ordinary web traffic. Its presence says "
        "nothing beyond that the client speaks HTTP-3.",
        {
            "quic_flow_count": len(flows),
            "total_flow_count": len(report.flows),
        },
        (SourceField.DST_PORT, SourceField.FLOW_ID, SourceField.PROTOCOL),
        [f.flow_id for f in flows],
    )


def _detect_upload_asymmetry(report: CaptureReport, cfg: SignalConfig) -> Signal | None:
    """Flows that sent substantially more than they received.

    Source: ``flows.bytes_out``, ``flows.bytes_in``.  Ordinary client traffic
    is download-dominant, so the reverse is worth noticing -- and no more.
    """
    flagged = [
        flow for flow in report.flows
        if flow.bytes_out >= cfg.upload_min_bytes_out
        and flow.bytes_out >= cfg.upload_min_ratio * max(flow.bytes_in, 1)
    ]
    if not flagged:
        return None

    total_out = sum(flow.bytes_out for flow in flagged)
    total_in = sum(flow.bytes_in for flow in flagged)
    ratio = round(total_out / max(total_in, 1), 4)

    return _make(
        SignalType.UPLOAD_ASYMMETRY,
        Severity.MEDIUM,
        _confidence(ratio, cfg.upload_min_ratio, cfg.upload_min_ratio * 10),
        f"{len(flagged)} flow(s) sent far more than they received "
        f"({total_out} bytes out against {total_in} in).",
        "Backups, file uploads, video calls and telemetry are all upload-heavy by "
        "design. Direction alone does not identify exfiltration.",
        {
            "flagged_flow_count": len(flagged),
            "bytes_out": total_out,
            "bytes_in": total_in,
            "out_in_ratio": ratio,
            "ratio_threshold": cfg.upload_min_ratio,
        },
        (SourceField.BYTES_IN, SourceField.BYTES_OUT, SourceField.FLOW_ID),
        [flow.flow_id for flow in flagged],
    )


def _detect_nonstandard_port_egress(report: CaptureReport, cfg: SignalConfig) -> Signal | None:
    """Flows to destination ports outside the documented common set.

    Source: ``flows.dst_port``, ``flows.bytes_out``, ``flows.bytes_in``.
    """
    flagged = [
        flow for flow in report.flows
        if flow.dst_port not in COMMON_PORTS
        and (flow.bytes_out + flow.bytes_in) >= cfg.nonstandard_min_bytes
    ]
    if len(flagged) < cfg.nonstandard_min_flows:
        return None

    ports = sorted({flow.dst_port for flow in flagged})
    return _make(
        SignalType.NONSTANDARD_PORT_EGRESS,
        Severity.LOW,
        _confidence(len(flagged), cfg.nonstandard_min_flows,
                    max(cfg.nonstandard_min_flows * 5, 10)),
        f"{len(flagged)} flow(s) carried traffic to {len(ports)} destination port(s) "
        "outside the common set.",
        "Peer-to-peer software, games, conferencing and VPNs use high dynamic ports "
        "routinely. A port being uncommon is not a property of the traffic's intent.",
        {
            "flagged_flow_count": len(flagged),
            "distinct_ports": ports[:_MAX_EVIDENCE_LIST],
            "min_bytes_threshold": cfg.nonstandard_min_bytes,
        },
        (SourceField.BYTES_IN, SourceField.BYTES_OUT, SourceField.DST_PORT,
         SourceField.FLOW_ID),
        [flow.flow_id for flow in flagged],
    )


def _detect_blocked_traffic(report: CaptureReport, cfg: SignalConfig) -> Signal | None:
    """Flows the engine's own rules acted on.

    Source: ``flows.verdict``, ``flows.state``, ``report.totals.dropped_packets``
    and ``report.blocking_rules_active``.
    """
    blocked = [f for f in report.flows if f.verdict == "DROP" or f.state == "BLOCKED"]
    dropped_packets = report.totals.dropped_packets
    rules_configured = sum(report.blocking_rules_active.values())

    if not blocked and dropped_packets <= 0:
        return None

    return _make(
        SignalType.BLOCKED_TRAFFIC_PRESENT,
        Severity.LOW,
        0.95,
        f"{len(blocked)} flow(s) were blocked by configured rules; "
        f"{dropped_packets} packet(s) were dropped.",
        "A block reflects the operator's own configuration. It records what a rule "
        "matched, never that the traffic was malicious.",
        {
            "blocked_flow_count": len(blocked),
            "dropped_packets": dropped_packets,
            "configured_rule_count": rules_configured,
        },
        (SourceField.BLOCKING_RULES_ACTIVE, SourceField.DROPPED_PACKETS,
         SourceField.FLOW_ID, SourceField.STATE, SourceField.VERDICT),
        [f.flow_id for f in blocked],
    )


def _detect_baseline_web_browsing(report: CaptureReport, cfg: SignalConfig) -> Signal | None:
    """How closely the capture resembles ordinary web browsing.

    Source: ``flows.dst_port``, ``flows.bytes_in``, ``flows.bytes_out``,
    ``flows.syn_ack_seen``, ``flows.server_name``.

    Emitted for every non-empty capture, with confidence reflecting the fit.
    That is deliberate: it gives the later retrieval step something other than
    alarm-shaped material to work with, which is what stops a corpus full of
    attack descriptions from making every capture look like an incident.
    """
    flows = report.flows
    if not flows:
        return None

    web = [f for f in flows if f.dst_port == _TLS_PORT or f.dst_port == _HTTP_PORT]
    named = [f for f in flows if f.server_name]
    download_heavy = [f for f in flows if f.bytes_in > f.bytes_out]
    completed = [f for f in flows
                 if f.protocol is TransportProtocol.TCP and f.syn_ack_seen]
    tcp_total = sum(1 for f in flows if f.protocol is TransportProtocol.TCP)

    web_share = _ratio(len(web), len(flows))
    named_share = _ratio(len(named), len(flows))
    download_share = _ratio(len(download_heavy), len(flows))
    completed_share = _ratio(len(completed), tcp_total)

    fit = round((web_share + named_share + download_share + completed_share) / 4, 4)

    return _make(
        SignalType.BASELINE_WEB_BROWSING,
        Severity.INFO,
        round(min(0.95, max(0.1, fit)), 4),
        f"{web_share:.0%} of flows target ports 80/443, {named_share:.0%} carry a "
        f"hostname and {download_share:.0%} are download-dominant.",
        "A high score describes the shape of ordinary browsing; it is not a "
        "certificate of safety, and a low score is not evidence of anything.",
        {
            "web_port_share": web_share,
            "named_flow_share": named_share,
            "download_dominant_share": download_share,
            "completed_handshake_share": completed_share,
            "browsing_fit": fit,
            "total_flow_count": len(flows),
        },
        (SourceField.BYTES_IN, SourceField.BYTES_OUT, SourceField.DST_PORT,
         SourceField.SERVER_NAME, SourceField.SYN_ACK_SEEN),
    )


#: Fixed evaluation order.  The output order is decided by the sort in
#: :func:`extract_signals`, so this list only affects readability.
DETECTORS: Final[tuple[Detector, ...]] = (
    _detect_unknown_app_share,
    _detect_dns_high_volume,
    _detect_dns_high_cardinality,
    _detect_dns_anomalous_label,
    _detect_scan_port_fanout,
    _detect_scan_half_open,
    _detect_tls_without_sni,
    _detect_plaintext_http,
    _detect_quic_present,
    _detect_upload_asymmetry,
    _detect_nonstandard_port_egress,
    _detect_blocked_traffic,
    _detect_baseline_web_browsing,
)


# ===========================================================================
# Profile and extraction
# ===========================================================================
def _build_profile(report: CaptureReport) -> CaptureProfile:
    """Distributions, straight from the report.  No thresholds, no judgement."""
    protocols: Counter[str] = Counter(flow.protocol.value for flow in report.flows)
    verdicts: Counter[str] = Counter(flow.verdict for flow in report.flows)
    states: Counter[str] = Counter(flow.state for flow in report.flows)
    ports: Counter[int] = Counter(flow.dst_port for flow in report.flows)

    return CaptureProfile(
        application_distribution=dict(
            sorted(report.application_distribution.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        protocol_distribution=dict(sorted(protocols.items(), key=lambda kv: (-kv[1], kv[0]))),
        verdict_distribution=dict(sorted(verdicts.items(), key=lambda kv: (-kv[1], kv[0]))),
        state_distribution=dict(sorted(states.items(), key=lambda kv: (-kv[1], kv[0]))),
        top_destination_ports=[
            (port, count)
            for port, count in sorted(ports.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        ],
    )


def extract_signals(
    report: CaptureReport, config: SignalConfig | None = None
) -> SignalReport:
    """Turn a sanitized capture report into a deterministic signal report.

    Pure: no clock, no randomness, no network, no model.  The same report
    always produces the same signals, in the same order, with the same ids and
    the same serialized bytes.

    Output order is **severity descending, then confidence descending, then
    signal type ascending** -- so the most relevant observations lead, and the
    order never depends on which detector happened to run first.
    """
    cfg = config or SignalConfig()

    signals: list[Signal] = []
    for detect in DETECTORS:
        signal = detect(report, cfg)
        if signal is not None:
            signals.append(signal)

    signals.sort(key=lambda s: (-_SEVERITY_RANK[s.severity],
                                -round(s.confidence, 6),
                                s.signal_type.value))

    known = report.flow_ids()
    problems: list[str] = []
    for signal in signals:
        problems.extend(signal.validate_flow_ids(frozenset(known)))
    if problems:  # pragma: no cover - only a detector bug can reach this
        raise ValueError("signal extraction produced unknown flow ids: " + "; ".join(problems))

    return SignalReport(
        generated_from=f"ai.schemas.CaptureReport v{report.schema_version}",
        capture_name=report.capture_name,
        redaction_mode=report.redaction_mode,
        flow_count=len(report.flows),
        total_flow_count=report.totals.total_flows,
        signal_count=len(signals),
        signals=tuple(signals),
        profile=_build_profile(report),
        thresholds={key: float(value) for key, value in cfg.as_dict().items()},
    )


# ===========================================================================
# Manual check:  python -m ai.rag.signals
# ===========================================================================
if __name__ == "__main__":  # pragma: no cover - manual check
    from ..schemas import CaptureTotals

    def flow(flow_id: int, **overrides) -> FlowRecord:
        defaults = dict(
            flow_id=flow_id, protocol=TransportProtocol.TCP, dst_port=443, src_port=50000,
            server_name="example.com", application="HTTPS", state="CLASSIFIED",
            verdict="FORWARD", packets_out=6, packets_in=9, bytes_out=900, bytes_in=4200,
            syn_seen=True, syn_ack_seen=True, fin_seen=True, src_ip="host-1", dst_ip="net-1",
        )
        defaults.update(overrides)
        return FlowRecord(**defaults)

    demo_flows = [flow(i) for i in range(4)]
    demo_flows += [
        flow(4 + i, protocol=TransportProtocol.UDP, dst_port=53, application="DNS",
             server_name=f"{'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6'[:20]}{i}.tunnel.example",
             syn_seen=False, syn_ack_seen=False, fin_seen=False,
             bytes_out=800, bytes_in=120)
        for i in range(6)
    ]
    demo_flows += [
        flow(10 + i, dst_port=9000 + i, application="Unknown", server_name=None,
             syn_ack_seen=False, fin_seen=False, bytes_out=64, bytes_in=0)
        for i in range(9)
    ]

    demo_report = CaptureReport(
        capture_name="synthetic.pcap",
        totals=CaptureTotals(
            total_packets=400, total_bytes=90_000, tcp_packets=300, udp_packets=100,
            forwarded_packets=390, dropped_packets=10,
            total_flows=len(demo_flows), flows_included=len(demo_flows),
        ),
        application_distribution={"HTTPS": 4, "DNS": 6, "Unknown": 9},
        top_server_names=["example.com"],
        blocking_rules_active={"blocked_ips": 2},
        flows=demo_flows,
        redaction_mode="redact_private",
        notes=[],
    )

    result = extract_signals(demo_report)
    print(f"capture:   {result.capture_name}  ({result.flow_count} flows)")
    print(f"schema:    {result.schema_version}   from {result.generated_from}")
    print(f"redaction: {result.redaction_mode}")
    print(f"signals:   {result.signal_count}\n")
    for item in result.signals:
        flows_note = (f"{len(item.flow_ids)} flow(s)" if item.flow_ids
                      else "capture-wide")
        print(f"  {item.severity.value:<6} {item.confidence:.2f}  "
              f"{item.signal_type.value:<24} {flows_note}")
        print(f"         {item.summary}")
        print(f"         not proven: {item.does_not_prove}")
    print(f"\nprotocols: {result.profile.protocol_distribution}")
    print(f"top ports: {result.profile.top_destination_ports[:5]}")
