"""Test runner for DPI signal extraction -- RAG step 5 only.

Scope
-----
Covers :mod:`ai.rag.signals`: the signal vocabulary, the models and their
validation, every detector's threshold behaviour, ordering, determinism,
provenance and privacy.  Retrieval, query building, embeddings and the vector
store are not involved and are asserted absent.

Dependencies
------------
Standard library and pydantic only.  **No** PCAP file, **no** API key, **no**
embedding model, **no** Hugging Face, **no** numpy and **no** vector store are
needed: every fixture is a synthetic :class:`~ai.schemas.CaptureReport` built
in memory, which is also what makes the "remove one flow" and "change
something unrelated" checks possible at all.

One optional check runs the real DPI engine over ``test_dpi.pcap`` if it is
present, purely to confirm the extractor works on genuine engine output.  It
skips cleanly when the capture is missing.

Run::

    python run_rag_signal_tests.py
"""

from __future__ import annotations

import io
import json
import os
import re
import socket
import sys
from contextlib import redirect_stdout
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
from ai.rag.documents import KNOWN_SIGNALS
from ai.rag.signals import (
    COMMON_PORTS,
    SIGNAL_SCHEMA_VERSION,
    CaptureProfile,
    Signal,
    SignalConfig,
    SignalReport,
    SignalType,
    SourceField,
    extract_signals,
    shannon_entropy,
)

_passed = 0
_failed = 0
_skipped = 0


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
# Synthetic fixtures
# ===========================================================================
def flow(flow_id: int, **overrides) -> FlowRecord:
    """A well-formed, unremarkable HTTPS flow, with fields overridable."""
    defaults = dict(
        flow_id=flow_id,
        protocol=TransportProtocol.TCP,
        dst_port=443,
        src_port=50000 + flow_id,
        server_name="www.example.com",
        application="HTTPS",
        state="CLASSIFIED",
        verdict="FORWARD",
        packets_out=8,
        packets_in=12,
        bytes_out=900,
        bytes_in=6400,
        syn_seen=True,
        syn_ack_seen=True,
        fin_seen=True,
        src_ip="host-1",
        dst_ip="net-1",
    )
    defaults.update(overrides)
    return FlowRecord(**defaults)


def dns_flow(flow_id: int, name: str, **overrides) -> FlowRecord:
    return flow(
        flow_id,
        protocol=TransportProtocol.UDP,
        dst_port=53,
        application="DNS",
        server_name=name,
        syn_seen=False,
        syn_ack_seen=False,
        fin_seen=False,
        packets_out=1,
        packets_in=1,
        bytes_out=80,
        bytes_in=180,
        **overrides,
    )


def report(
    flows: list[FlowRecord],
    *,
    dropped_packets: int = 0,
    rules: dict[str, int] | None = None,
    applications: dict[str, int] | None = None,
    redaction_mode: str = "redact_private",
    capture_name: str = "synthetic.pcap",
    total_flows: int | None = None,
) -> CaptureReport:
    """Build a CaptureReport whose totals agree with its flows."""
    if applications is None:
        applications = {}
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
            forwarded_packets=sum(f.packets_out + f.packets_in for f in flows
                                  if f.verdict == "FORWARD"),
            dropped_packets=dropped_packets,
            total_flows=total_flows if total_flows is not None else len(flows),
            flows_included=len(flows),
        ),
        application_distribution=applications,
        top_server_names=sorted({f.server_name for f in flows if f.server_name})[:5],
        blocking_rules_active=rules or {},
        flows=flows,
        redaction_mode=redaction_mode,
        notes=[],
    )


def browsing_report(count: int = 8) -> CaptureReport:
    """A plain, well-behaved browsing capture: no signal should be alarming."""
    return report([flow(i, server_name=f"host{i}.example.com") for i in range(count)])


def sample_signal(**overrides) -> Signal:
    """A valid Signal, for the model rejection cases."""
    fields = dict(
        signal_id="quic_present#0123456789ab",
        signal_type=SignalType.QUIC_PRESENT,
        severity=Severity.INFO,
        confidence=0.9,
        summary="Three flows used UDP port 443, consistent with QUIC.",
        does_not_prove="QUIC is ordinary web traffic and implies nothing further.",
        evidence={"quic_flow_count": 3},
        flow_ids=(1, 2, 3),
        source_fields=(SourceField.DST_PORT, SourceField.PROTOCOL),
    )
    fields.update(overrides)
    return Signal(**fields)


# ===========================================================================
# 1. Vocabulary and contract with the corpus
# ===========================================================================
def test_vocabulary() -> None:
    print("\nVocabulary")

    check("SignalType matches the corpus signal vocabulary exactly",
          {t.value for t in SignalType} == set(KNOWN_SIGNALS),
          str(sorted({t.value for t in SignalType} ^ set(KNOWN_SIGNALS))))
    check("every signal type is lowercase snake_case",
          all(re.fullmatch(r"[a-z][a-z0-9_]*", t.value) for t in SignalType))
    check("every source field names a real report or flow path",
          all(f.value.startswith(("report.", "flows.")) for f in SourceField))

    # Every SourceField must exist on the models it claims to come from.
    flow_fields = set(FlowRecord.model_fields)
    report_fields = set(CaptureReport.model_fields)
    totals_fields = set(CaptureTotals.model_fields)
    for field in SourceField:
        path = field.value
        if path.startswith("flows."):
            ok = path.split(".", 1)[1] in flow_fields
        elif path.startswith("report.totals."):
            ok = path.rsplit(".", 1)[1] in totals_fields
        else:
            ok = path.split(".", 1)[1] in report_fields
        check(f"source field {path} exists on the real schema", ok)

    # No signal is claimed that the schema cannot support.
    for absent in ("beaconing", "duration", "jitter", "packet_rate", "ja3",
                   "repeated_destination"):
        check(f"no {absent!r} signal is claimed",
              not any(absent in t.value for t in SignalType))


# ===========================================================================
# 2. Empty and minimal captures
# ===========================================================================
def test_empty_capture() -> None:
    print("\nEmpty capture")

    empty = extract_signals(report([]))
    check("an empty capture produces a valid report", isinstance(empty, SignalReport))
    check("an empty capture produces no signals", empty.signal_count == 0)
    check("signals tuple is empty", empty.signals == ())
    check("flow_count is zero", empty.flow_count == 0)
    check("the report still records its schema version",
          empty.schema_version == SIGNAL_SCHEMA_VERSION == "1.0")
    check("the report still records what it came from",
          "CaptureReport" in empty.generated_from)
    check("the report still records the redaction mode",
          empty.redaction_mode == "redact_private")
    check("an empty capture still serialises", json.loads(empty.to_json())["signal_count"] == 0)
    check("an empty capture has an empty profile",
          empty.profile.protocol_distribution == {})
    check("by_type finds nothing on an empty report",
          empty.by_type(SignalType.QUIC_PRESENT) is None)


# ===========================================================================
# 3. Determinism
# ===========================================================================
def test_determinism() -> None:
    print("\nDeterminism")

    source = report(
        [flow(i) for i in range(4)]
        + [dns_flow(4 + i, f"name{i}.example.com") for i in range(6)]
        + [flow(10 + i, dst_port=9000 + i, application="Unknown", server_name=None,
                syn_ack_seen=False, bytes_out=800, bytes_in=0) for i in range(5)]
    )

    first = extract_signals(source)
    second = extract_signals(source)

    check("signal ids are identical across runs",
          [s.signal_id for s in first.signals] == [s.signal_id for s in second.signals])
    check("signal ordering is identical across runs", first.types() == second.types())
    check("evidence is identical across runs",
          [s.evidence for s in first.signals] == [s.evidence for s in second.signals])
    check("flow ids are identical across runs",
          [s.flow_ids for s in first.signals] == [s.flow_ids for s in second.signals])
    check("serialization is byte-identical across runs", first.to_json() == second.to_json())
    check("an equivalent report rebuilt from scratch serialises identically",
          extract_signals(report(list(source.flows))).to_json() == first.to_json())

    # The documented ordering rule.
    rank = {Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2, Severity.HIGH: 3}
    keys = [(-rank[s.severity], -round(s.confidence, 6), s.signal_type.value)
            for s in first.signals]
    check("signals are ordered by severity, then confidence, then type",
          keys == sorted(keys), str(first.types()))

    check("flow ids inside each signal are sorted",
          all(list(s.flow_ids) == sorted(s.flow_ids) for s in first.signals))
    check("flow ids inside each signal are unique",
          all(len(set(s.flow_ids)) == len(s.flow_ids) for s in first.signals))
    check("source fields are sorted",
          all([f.value for f in s.source_fields] == sorted(f.value for f in s.source_fields)
              for s in first.signals))
    check("signal ids are content-derived, not positional",
          all(s.signal_id.startswith(f"{s.signal_type.value}#") and
              re.fullmatch(r"[0-9a-f]{12}", s.signal_id.rsplit("#", 1)[1])
              for s in first.signals))

    # A different capture must produce different ids for the same signal type.
    other = extract_signals(report(
        [dns_flow(i, f"other{i}.example.com") for i in range(7)]))
    same_type = {s.signal_type for s in first.signals} & {s.signal_type for s in other.signals}
    check("the same signal type over different data gets a different id",
          all(first.by_type(t).signal_id != other.by_type(t).signal_id
              for t in same_type if first.by_type(t).evidence != other.by_type(t).evidence),
          str(sorted(t.value for t in same_type)))


# ===========================================================================
# 4. Model validation
# ===========================================================================
def test_models() -> None:
    print("\nModel validation")

    check("a well-formed signal validates", isinstance(sample_signal(), Signal))
    check("signals are immutable", Signal.model_config["frozen"] is True)
    check("signal reports are immutable", SignalReport.model_config["frozen"] is True)

    raises("an unknown signal type is rejected", ValueError,
           lambda: sample_signal(signal_type="totally_made_up"))
    raises("an invalid severity is rejected", ValueError,
           lambda: sample_signal(severity="catastrophic"))
    raises("confidence below zero is rejected", ValueError,
           lambda: sample_signal(confidence=-0.1))
    raises("confidence above one is rejected", ValueError,
           lambda: sample_signal(confidence=1.1))
    raises("an unknown source field is rejected", ValueError,
           lambda: sample_signal(source_fields=("flows.made_up_field",)))
    raises("an arbitrary source string is rejected", ValueError,
           lambda: sample_signal(source_fields=("whatever I like",)))
    raises("an empty source field list is rejected", ValueError,
           lambda: sample_signal(source_fields=()))
    raises("empty evidence is rejected", ValueError, lambda: sample_signal(evidence={}))
    raises("an evidence key that is not snake_case is rejected", ValueError,
           lambda: sample_signal(evidence={"Not Snake Case": 1}))
    raises("raw bytes in evidence are rejected", ValueError,
           lambda: sample_signal(evidence={"payload": b"\x16\x03\x01"}))
    raises("an over-long evidence string is rejected", ValueError,
           lambda: sample_signal(evidence={"blob": "x" * 500}))
    raises("an over-long evidence list is rejected", ValueError,
           lambda: sample_signal(evidence={"ids": list(range(500))}))
    raises("an empty signal id is rejected", ValueError, lambda: sample_signal(signal_id=""))
    raises("a malformed signal id is rejected", ValueError,
           lambda: sample_signal(signal_id="not-an-id"))
    raises("a signal id that disagrees with its type is rejected", ValueError,
           lambda: sample_signal(signal_id="dns_high_volume#0123456789ab"))
    raises("unsorted flow ids are rejected", ValueError,
           lambda: sample_signal(flow_ids=(3, 1, 2)))
    raises("duplicate flow ids are rejected", ValueError,
           lambda: sample_signal(flow_ids=(1, 1, 2)))
    raises("negative flow ids are rejected", ValueError,
           lambda: sample_signal(flow_ids=(-1, 2)))
    raises("an unexpected field on a signal is rejected", ValueError,
           lambda: sample_signal(unexpected="value"))
    raises("an empty summary is rejected", ValueError, lambda: sample_signal(summary=""))
    raises("a missing does_not_prove is rejected", ValueError,
           lambda: sample_signal(does_not_prove=""))

    check("a capture-wide signal may carry no flow ids",
          sample_signal(flow_ids=()).flow_ids == ())

    # -- invented flow ids -------------------------------------------------
    signal = sample_signal(flow_ids=(1, 2, 99))
    problems = signal.validate_flow_ids(frozenset({1, 2, 3}))
    check("a signal citing an unknown flow id is detected", problems != [])
    check("the problem names the offending id", "99" in problems[0], problems[0])
    check("a signal citing only real flow ids is clean",
          sample_signal(flow_ids=(1, 2)).validate_flow_ids(frozenset({1, 2, 3})) == [])

    real = extract_signals(browsing_report())
    check("extracted signals never cite an unknown flow id",
          real.validate_flow_ids(frozenset(real_ids(real))) == [])

    # -- report-level rules ------------------------------------------------
    raises("a signal_count that disagrees with the signals is rejected", ValueError,
           lambda: SignalReport(generated_from="x", capture_name="c",
                                redaction_mode="none", flow_count=0,
                                total_flow_count=0, signal_count=5,
                                signals=(sample_signal(),)))
    raises("a duplicated signal type in one report is rejected", ValueError,
           lambda: SignalReport(generated_from="x", capture_name="c",
                                redaction_mode="none", flow_count=0,
                                total_flow_count=0, signal_count=2,
                                signals=(sample_signal(), sample_signal())))
    raises("out-of-order signals are rejected", ValueError,
           lambda: SignalReport(
               generated_from="x", capture_name="c", redaction_mode="none",
               flow_count=0, total_flow_count=0, signal_count=2,
               signals=(sample_signal(),
                        sample_signal(signal_id="plaintext_http#0123456789ab",
                                      signal_type=SignalType.PLAINTEXT_HTTP,
                                      severity=Severity.LOW))))


def real_ids(signal_report: SignalReport) -> set[int]:
    """Every flow id any signal in the report cites."""
    ids: set[int] = set()
    for signal in signal_report.signals:
        ids.update(signal.flow_ids)
    return ids


# ===========================================================================
# 5. Detector behaviour
# ===========================================================================
def test_detectors() -> None:
    print("\nDetectors")

    types = lambda r: set(extract_signals(r).types())  # noqa: E731

    # -- unknown_app_share -------------------------------------------------
    clean = browsing_report()
    check("a capture with no unknown flows produces no unknown_app_share signal",
          "unknown_app_share" not in types(clean), str(types(clean)))

    unknown_flows = [flow(i, application="Unknown", server_name=None) for i in range(5)]
    mixed = report(unknown_flows + [flow(5 + i) for i in range(5)])
    result = extract_signals(mixed)
    signal = result.by_type(SignalType.UNKNOWN_APP_SHARE)
    check("unknown flows produce an unknown_app_share signal", signal is not None)
    check("its evidence matches the actual unknown flow count",
          signal.evidence["unknown_flow_count"] == 5, str(signal.evidence))
    check("its evidence matches the actual total flow count",
          signal.evidence["total_flow_count"] == 10, str(signal.evidence))
    check("its ratio is the real ratio", signal.evidence["ratio"] == 0.5)
    check("it cites exactly the unknown flows",
          signal.flow_ids == (0, 1, 2, 3, 4), str(signal.flow_ids))
    check("it cites flows.application as a source",
          SourceField.APPLICATION in signal.source_fields)
    check("it does not claim maliciousness",
          "malicious" not in signal.summary.lower()
          and "risk" in signal.does_not_prove.lower())

    # -- DNS ---------------------------------------------------------------
    check("a capture with no DNS produces no DNS signals",
          not any(t.startswith("dns_") for t in types(clean)), str(types(clean)))

    dns_report = report([dns_flow(i, f"unique{i}.example.com") for i in range(8)])
    dns_types = types(dns_report)
    check("DNS flows produce a dns_high_volume signal", "dns_high_volume" in dns_types)
    check("distinct DNS names produce a dns_high_cardinality signal",
          "dns_high_cardinality" in dns_types)
    volume = extract_signals(dns_report).by_type(SignalType.DNS_HIGH_VOLUME)
    check("the DNS evidence matches the actual DNS flow count",
          volume.evidence["dns_flow_count"] == 8, str(volume.evidence))
    check("the DNS signal cites every DNS flow", len(volume.flow_ids) == 8)

    repeated = report([dns_flow(i, "www.example.com") for i in range(8)])
    check("repeated DNS names produce no cardinality signal",
          "dns_high_cardinality" not in types(repeated), str(types(repeated)))

    check("ordinary DNS names produce no anomalous-label signal",
          "dns_anomalous_label" not in types(dns_report))
    tunnelish = report([
        dns_flow(i, f"{'k7f2q9x4m1z8b3v6n5c0'}{i}.tunnel.example") for i in range(6)
    ])
    check("long high-entropy labels produce a dns_anomalous_label signal",
          "dns_anomalous_label" in types(tunnelish), str(types(tunnelish)))

    # -- scanning ----------------------------------------------------------
    check("ordinary traffic produces no fan-out signal",
          "scan_port_fanout" not in types(clean))
    fanout = report([flow(i, dst_port=9000 + i, server_name=None,
                          application="Unknown", bytes_out=64, bytes_in=0)
                     for i in range(10)])
    check("many distinct destination ports produce scan_port_fanout",
          "scan_port_fanout" in types(fanout), str(types(fanout)))
    fan_signal = extract_signals(fanout).by_type(SignalType.SCAN_PORT_FANOUT)
    check("its evidence reports the real distinct-port count",
          fan_signal.evidence["distinct_destination_ports"] == 10,
          str(fan_signal.evidence))
    check("it records how flows were grouped",
          fan_signal.evidence["grouped_by"] == "src_ip")

    check("completed handshakes produce no half-open signal",
          "scan_half_open" not in types(clean))
    half = report([flow(i, syn_ack_seen=False, fin_seen=False) for i in range(6)])
    check("unanswered SYNs produce a scan_half_open signal",
          "scan_half_open" in types(half))
    half_signal = extract_signals(half).by_type(SignalType.SCAN_HALF_OPEN)
    check("its evidence matches the real half-open count",
          half_signal.evidence["half_open_count"] == 6 and
          half_signal.evidence["ratio"] == 1.0, str(half_signal.evidence))

    # -- classification signals -------------------------------------------
    check("named TLS flows produce no tls_without_sni signal",
          "tls_without_sni" not in types(clean))
    check("unnamed TLS flows produce a tls_without_sni signal",
          "tls_without_sni" in types(report([flow(0, server_name=None)])))

    check("HTTPS-only traffic produces no plaintext_http signal",
          "plaintext_http" not in types(clean))
    check("port 80 traffic produces a plaintext_http signal",
          "plaintext_http" in types(report([flow(0, dst_port=80, application="HTTP")])))

    check("TCP-only traffic produces no QUIC signal", "quic_present" not in types(clean))
    check("UDP 443 traffic produces a quic_present signal",
          "quic_present" in types(report([
              flow(0, protocol=TransportProtocol.UDP, dst_port=443, application="QUIC",
                   syn_seen=False, syn_ack_seen=False, fin_seen=False)])))

    # -- volume and ports --------------------------------------------------
    check("download-dominant traffic produces no upload_asymmetry signal",
          "upload_asymmetry" not in types(clean))
    check("upload-dominant traffic produces an upload_asymmetry signal",
          "upload_asymmetry" in types(report([flow(0, bytes_out=200_000, bytes_in=1000)])))

    check("common ports produce no nonstandard_port_egress signal",
          "nonstandard_port_egress" not in types(clean))
    check("uncommon ports with real traffic produce nonstandard_port_egress",
          "nonstandard_port_egress" in types(report(
              [flow(i, dst_port=44300 + i, bytes_out=5000, bytes_in=5000)
               for i in range(3)])))
    check("uncommon ports with negligible traffic do not",
          "nonstandard_port_egress" not in types(report(
              [flow(i, dst_port=44300 + i, bytes_out=1, bytes_in=0) for i in range(3)])),
          "the byte floor should suppress this")
    check("every COMMON_PORTS entry is a valid port number",
          all(0 < port <= 65535 for port in COMMON_PORTS))

    # -- blocked traffic ---------------------------------------------------
    check("a capture with no blocks produces no blocked_traffic signal",
          "blocked_traffic_present" not in types(clean), str(types(clean)))
    check("a dropped verdict produces a blocked_traffic signal",
          "blocked_traffic_present" in types(report(
              [flow(0, verdict="DROP", state="BLOCKED")])))
    check("dropped packets alone produce a blocked_traffic signal",
          "blocked_traffic_present" in types(report([flow(0)], dropped_packets=12)))
    blocked = extract_signals(report([flow(0, verdict="DROP", state="BLOCKED"), flow(1)],
                                     dropped_packets=12,
                                     rules={"blocked_ips": 2, "blocked_ports": 1}))
    blocked_signal = blocked.by_type(SignalType.BLOCKED_TRAFFIC_PRESENT)
    check("its evidence matches the real blocked flow count",
          blocked_signal.evidence["blocked_flow_count"] == 1, str(blocked_signal.evidence))
    check("its evidence matches the real dropped packet count",
          blocked_signal.evidence["dropped_packets"] == 12)
    check("its evidence counts the configured rules",
          blocked_signal.evidence["configured_rule_count"] == 3)
    check("it does not call blocked traffic malicious",
          "malicious" in blocked_signal.does_not_prove.lower())

    # -- baseline ----------------------------------------------------------
    baseline = extract_signals(clean).by_type(SignalType.BASELINE_WEB_BROWSING)
    check("ordinary browsing produces a baseline signal", baseline is not None)
    check("the baseline signal is informational", baseline.severity is Severity.INFO)
    check("the baseline signal is capture-wide", baseline.flow_ids == ())
    check("the baseline signal scores a browsing capture highly",
          baseline.evidence["browsing_fit"] >= 0.7, str(baseline.evidence))
    check("the baseline signal scores a scan capture lowly",
          extract_signals(fanout).by_type(
              SignalType.BASELINE_WEB_BROWSING).evidence["browsing_fit"] <= 0.3)
    check("the baseline signal does not claim safety",
          "safety" in baseline.does_not_prove.lower()
          or "certificate of safety" in baseline.does_not_prove.lower())

    # Every signal must explain what it does not prove.
    everything = extract_signals(report(
        [flow(i, dst_port=80, application="HTTP") for i in range(2)]
        + [dns_flow(2 + i, f"n{i}.example.com") for i in range(6)]))
    check("every emitted signal explains what it does not prove",
          all(len(s.does_not_prove) >= 10 for s in everything.signals))
    check("no summary claims maliciousness",
          not any("malicious" in s.summary.lower() for s in everything.signals))


# ===========================================================================
# 6. Thresholds
# ===========================================================================
def test_thresholds() -> None:
    print("\nThresholds")

    default = SignalConfig()
    check("thresholds are exposed for audit", "unknown_app_min_share" in default.as_dict())
    check("the config is immutable",
          type(default).__dataclass_params__.frozen)  # type: ignore[attr-defined]
    check("every threshold is recorded in the report",
          set(extract_signals(browsing_report()).thresholds) == set(default.as_dict()))

    # Just below and just above the unknown-share threshold.
    below = report([flow(i, application="Unknown", server_name=None) for i in range(2)]
                   + [flow(2 + i) for i in range(8)])
    above = report([flow(i, application="Unknown", server_name=None) for i in range(3)]
                   + [flow(3 + i) for i in range(7)])
    check("below the count floor, no unknown_app_share signal fires",
          "unknown_app_share" not in extract_signals(below).types(),
          str(extract_signals(below).types()))
    check("at the count floor, the unknown_app_share signal fires",
          "unknown_app_share" in extract_signals(above).types())

    # A ratio alone must not fire it on a tiny capture.
    tiny = report([flow(0, application="Unknown", server_name=None), flow(1)])
    check("a 50% share of two flows does not fire the signal",
          "unknown_app_share" not in extract_signals(tiny).types(),
          "the absolute floor should suppress this")

    # Lowering the threshold must change the outcome, deterministically.
    lenient = SignalConfig(unknown_app_min_flows=1, unknown_app_min_share=0.1)
    check("lowering the threshold makes the signal fire",
          "unknown_app_share" in extract_signals(tiny, lenient).types())
    check("a custom configuration is still deterministic",
          extract_signals(tiny, lenient).to_json() == extract_signals(tiny, lenient).to_json())
    check("the custom threshold is recorded in the report",
          extract_signals(tiny, lenient).thresholds["unknown_app_min_flows"] == 1.0)

    strict = SignalConfig(dns_min_flows=100)
    dns_report = report([dns_flow(i, f"n{i}.example.com") for i in range(8)])
    check("raising a threshold suppresses the signal",
          "dns_high_volume" not in extract_signals(dns_report, strict).types())
    check("the default configuration still fires it",
          "dns_high_volume" in extract_signals(dns_report).types())

    check("entropy is computed, not guessed",
          abs(shannon_entropy("aaaa")) < 1e-12
          and abs(shannon_entropy("abcd") - 2.0) < 1e-12,
          f"{shannon_entropy('aaaa')} {shannon_entropy('abcd')}")
    check("entropy of an empty string is zero", shannon_entropy("") == 0.0)


# ===========================================================================
# 7. Locality: changing one flow changes only what depends on it
# ===========================================================================
def test_locality() -> None:
    print("\nLocality of change")

    flows = ([flow(i) for i in range(4)]
             + [dns_flow(4 + i, f"n{i}.example.com") for i in range(6)]
             + [flow(10 + i, dst_port=80, application="HTTP") for i in range(2)])
    before = extract_signals(report(flows))

    # Remove one HTTP flow: only the HTTP signal and the capture-wide
    # baseline may change.
    without_http = extract_signals(report([f for f in flows if f.flow_id != 11]))
    changed = {
        signal.signal_type.value
        for signal in without_http.signals
        if before.by_type(signal.signal_type) is None
        or before.by_type(signal.signal_type).evidence != signal.evidence
    }
    check("removing one flow changes the signal that cites it",
          "plaintext_http" in changed, str(sorted(changed)))
    dns_before = before.by_type(SignalType.DNS_HIGH_VOLUME).evidence
    dns_after = without_http.by_type(SignalType.DNS_HIGH_VOLUME).evidence
    check("the DNS signal's own count is unchanged by an unrelated removal",
          dns_after["dns_flow_count"] == dns_before["dns_flow_count"] == 6,
          f"{dns_before} -> {dns_after}")
    check("only the capture-wide total shifts",
          dns_after["total_flow_count"] == dns_before["total_flow_count"] - 1,
          f"{dns_before} -> {dns_after}")
    check("the DNS signal still cites exactly the DNS flows",
          without_http.by_type(SignalType.DNS_HIGH_VOLUME).flow_ids
          == before.by_type(SignalType.DNS_HIGH_VOLUME).flow_ids)
    check("the DNS cardinality evidence is unchanged by an unrelated removal",
          before.by_type(SignalType.DNS_HIGH_CARDINALITY).evidence["distinct_name_count"]
          == without_http.by_type(
              SignalType.DNS_HIGH_CARDINALITY).evidence["distinct_name_count"])

    # Change something genuinely unrelated: the capture name.
    renamed = extract_signals(report(flows, capture_name="other.pcap"))
    check("renaming the capture changes no signal id",
          [s.signal_id for s in renamed.signals] == [s.signal_id for s in before.signals])
    check("renaming the capture is reflected in the report",
          renamed.capture_name == "other.pcap")

    # Adding blocking rules must not disturb the DNS signals.
    with_rules = extract_signals(report(flows, dropped_packets=5,
                                        rules={"blocked_ips": 1}))
    check("adding blocked traffic introduces exactly one new signal",
          set(with_rules.types()) - set(before.types()) == {"blocked_traffic_present"},
          str(set(with_rules.types()) - set(before.types())))
    check("adding blocked traffic leaves the DNS signal id unchanged",
          with_rules.by_type(SignalType.DNS_HIGH_VOLUME).signal_id
          == before.by_type(SignalType.DNS_HIGH_VOLUME).signal_id)
    check("adding blocked traffic leaves the HTTP signal id unchanged",
          with_rules.by_type(SignalType.PLAINTEXT_HTTP).signal_id
          == before.by_type(SignalType.PLAINTEXT_HTTP).signal_id)


# ===========================================================================
# 8. Profile
# ===========================================================================
def test_profile() -> None:
    print("\nCapture profile")

    flows = ([flow(i) for i in range(3)]
             + [dns_flow(3 + i, f"n{i}.example.com") for i in range(2)]
             + [flow(5, verdict="DROP", state="BLOCKED")])
    source = report(flows)
    profile = extract_signals(source).profile

    check("the profile is a validated model", isinstance(profile, CaptureProfile))
    check("the application distribution matches the report",
          profile.application_distribution == dict(
              sorted(source.application_distribution.items(), key=lambda kv: (-kv[1], kv[0]))),
          str(profile.application_distribution))
    check("the protocol distribution matches the flows",
          profile.protocol_distribution == {"TCP": 4, "UDP": 2},
          str(profile.protocol_distribution))
    check("the verdict distribution matches the flows",
          profile.verdict_distribution == {"FORWARD": 5, "DROP": 1},
          str(profile.verdict_distribution))
    check("the state distribution matches the flows",
          profile.state_distribution == {"CLASSIFIED": 5, "BLOCKED": 1},
          str(profile.state_distribution))
    check("the top destination ports match the flows",
          profile.top_destination_ports[0] == (443, 4),
          str(profile.top_destination_ports))
    check("the profile is deterministic",
          extract_signals(source).profile == profile)
    check("flow_count and total_flow_count come from the report",
          extract_signals(source).flow_count == 6
          and extract_signals(report(flows, total_flows=99)).total_flow_count == 99)


# ===========================================================================
# 9. Privacy and isolation
# ===========================================================================
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def test_privacy() -> None:
    print("\nPrivacy and isolation")

    source_code = Path("ai/rag/signals.py").read_text(encoding="utf-8")

    # -- no clock, no randomness ------------------------------------------
    for banned in ("import time", "import random", "import uuid", "datetime",
                   "time.time", "monotonic", "uuid4"):
        check(f"the extractor never uses {banned!r}", banned not in source_code)
    check("the report model has no generated-at field",
          not any("generat" in name and name != "generated_from"
                  for name in SignalReport.model_fields))
    check("no timestamp key appears in a serialised report",
          not any(key in extract_signals(browsing_report()).to_json().lower()
                  for key in ("timestamp", "generated_at", "created_at")))

    # -- no downstream RAG code, no providers ------------------------------
    for banned in ("vector_store", "VectorStore", "embeddings", "EmbeddingModel",
                   "openai", "groq", "requests", "urllib", "langchain"):
        check(f"the extractor never references {banned!r}", banned not in source_code)
    # An AST audit rather than a substring search: the module's own docstring
    # mentions FlowSnapshot and dpi/ by name to explain why it does *not* use
    # them, and a naive grep would flag that prose.
    import ast

    imported: set[str] = set()
    for node in ast.walk(ast.parse(source_code)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
            if node.level:  # relative import, e.g. "from ..schemas import ..."
                imported.add(f".{node.module or ''}")

    check("the extractor imports nothing from the DPI package",
          "dpi" not in imported, str(sorted(imported)))
    check("the extractor imports no vector-store or embedding module",
          not any(name in {".vector_store", ".embeddings", "vector_store", "embeddings"}
                  for name in imported), str(sorted(imported)))
    check("the extractor's only project imports are the schema and the vocabulary",
          {name for name in imported if name.startswith(".")}
          <= {".schemas", ".documents"}, str(sorted(imported)))
    check("the extractor reads the sanitized report type",
          "CaptureReport" in source_code)
    check("FlowSnapshot is only mentioned in prose, never used",
          "FlowSnapshot" not in "".join(
              line for line in source_code.splitlines()
              if not line.lstrip().startswith("#")
              and "FlowSnapshot" in line and "dpi.dpi_engine" not in line))
    check("the openai SDK is not imported", "openai" not in sys.modules)
    check("no vector database library is imported",
          not any(name in sys.modules for name in ("faiss", "chromadb", "qdrant_client")))
    check("no LangChain library is imported",
          not any(name.startswith("langchain") for name in sys.modules))

    # -- the DPI engine stays independent of ai/rag ------------------------
    offenders = [path.name for path in Path("dpi").glob("*.py")
                 if "ai.rag" in path.read_text(encoding="utf-8")
                 or "from ai" in path.read_text(encoding="utf-8")]
    check("no DPI module imports anything from ai/", not offenders, str(offenders))

    # -- no payloads, no secrets ------------------------------------------
    os.environ["DPI_SIGNAL_TEST_SECRET"] = "sk-must-never-appear"
    try:
        rich = extract_signals(report(
            [flow(i, dst_port=44300 + i, bytes_out=200_000, bytes_in=10) for i in range(3)]
            + [dns_flow(3 + i, f"k7f2q9x4m1z8b3v6n5c{i}.tunnel.example") for i in range(6)]
            + [flow(9, verdict="DROP", state="BLOCKED")],
            dropped_packets=4, rules={"blocked_ips": 1}))
        serialized = rich.to_json()
        check("no environment value leaks into a report",
              "sk-must-never-appear" not in serialized)
        for marker in ("sk-", "api_key", "API_KEY", "Bearer ", "password", "payload"):
            check(f"no {marker!r} appears in a serialised report", marker not in serialized)
    finally:
        os.environ.pop("DPI_SIGNAL_TEST_SECRET", None)

    check("evidence values are JSON-native only",
          all(isinstance(value, (int, float, str, bool, list))
              for signal in rich.signals for value in signal.evidence.values()))
    check("no evidence value is raw bytes",
          not any(isinstance(value, (bytes, bytearray))
                  for signal in rich.signals for value in signal.evidence.values()))

    # -- addresses --------------------------------------------------------
    check("no IPv4 address appears anywhere in a report built from pseudonyms",
          not IPV4.search(serialized), str(IPV4.findall(serialized)[:3]))

    withheld = extract_signals(report(
        [flow(i, dst_port=9000 + i, src_ip=None, dst_ip=None, server_name=None,
              application="Unknown", bytes_out=64, bytes_in=0) for i in range(10)],
        redaction_mode="none"))
    fanout = withheld.by_type(SignalType.SCAN_PORT_FANOUT)
    check("fan-out still works when addresses are withheld", fanout is not None)
    check("it says so rather than pretending to know the host",
          fanout.evidence["grouped_by"] == "capture", str(fanout.evidence))
    check("it records the redaction mode that produced it",
          fanout.evidence["redaction_mode"] == "none")
    check("the report carries the redaction mode forward",
          withheld.redaction_mode == "none")
    check("no address appears in the withheld-mode report",
          not IPV4.search(withheld.to_json()))

    # -- no network -------------------------------------------------------
    real_socket, real_connect = socket.socket, socket.create_connection

    def refuse(*args, **kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("signal extraction attempted a network connection")

    socket.socket, socket.create_connection = refuse, refuse  # type: ignore[assignment]
    try:
        offline = extract_signals(browsing_report())
        check("signal extraction makes no network call", offline.signal_count >= 1)
    finally:
        socket.socket, socket.create_connection = real_socket, real_connect


# ===========================================================================
# 10. Optional: the real DPI engine
# ===========================================================================
def test_real_engine() -> None:
    print("\nOptional -- real DPI engine output")

    capture = Path("test_dpi.pcap")
    if not capture.is_file():
        skip("signals extract from a real DPI capture", "test_dpi.pcap is not present")
        return

    from ai.config import AIConfig
    from ai.extractor import build_capture_report
    from dpi.dpi_engine import Config, DPIEngine

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        engine = DPIEngine(Config())
        engine.initialize()
        engine.process_file(str(capture), os.devnull)
        snapshot = engine.get_flow_snapshot()

    real_report = build_capture_report(snapshot, str(capture), AIConfig())
    result = extract_signals(real_report)

    check("a real capture produces a valid signal report",
          isinstance(result, SignalReport))
    check("every signal cites only real flow ids",
          result.validate_flow_ids(frozenset(real_report.flow_ids())) == [])
    check("the report records the real capture name",
          result.capture_name == capture.name)
    check("the report records the real redaction mode",
          result.redaction_mode == real_report.redaction_mode)
    check("extraction from real engine output is deterministic",
          extract_signals(real_report).to_json() == result.to_json())
    check("no IPv4 address leaks from a real capture",
          not IPV4.search(result.to_json()), str(IPV4.findall(result.to_json())[:3]))
    print(f"        {real_report.totals.flows_included} flows -> "
          f"{result.signal_count} signal(s): {', '.join(result.types()) or 'none'}")


# ===========================================================================
def main() -> int:
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print("RAG step 5 -- DPI signal extraction")

    test_vocabulary()
    test_empty_capture()
    test_determinism()
    test_models()
    test_detectors()
    test_thresholds()
    test_locality()
    test_profile()
    test_privacy()
    test_real_engine()

    total = _passed + _failed
    suffix = f", {_skipped} skipped" if _skipped else ""
    print(f"\n{_passed}/{total} checks passed{suffix}")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
