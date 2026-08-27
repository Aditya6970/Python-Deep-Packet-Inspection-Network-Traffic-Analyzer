"""The evaluation dataset: captures paired with what should happen to them.

Where the facts come from
-------------------------
Two sources, and every case says which it used:

* ``pcap`` -- the real ``test_dpi.pcap``, run through the unmodified DPI engine
  and :func:`ai.extractor.build_capture_report`. Nothing is asserted about it
  that the engine did not measure.
* ``synthetic`` -- flows constructed field by field through
  :class:`~ai.schemas.FlowRecord`, which validates every one of them. These are
  *explicitly constructed test signals*: a capture built to contain a
  particular observable pattern, not invented packet facts. Counters are
  derived from the flows rather than typed in, so a report can never claim more
  packets than its flows contain.

Where the labels come from
--------------------------
By hand, from the corpus and the DPI schema, before any retrieval was run. A
label edited to match an observed ranking measures nothing, so the expectations
here are deliberately written from what a knowledgeable reader would say the
right answer is -- and where the system disagrees, that is a finding to report,
not a label to adjust.

Negatives are conservative. ``irrelevant_documents`` names only documents whose
appearance in the top results would be a real error -- an attack description
retrieved for plainly ordinary browsing, say. Everything unlisted is
*unjudged*, which is the honest state for most of a corpus.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final

from ai.schemas import CaptureReport, CaptureTotals, FlowRecord, TransportProtocol

__all__ = [
    "CASES",
    "CORPUS_DOCUMENT_IDS",
    "EvaluationCase",
    "build_report",
    "cases_for_live",
    "pcap_available",
]

#: The six documents in ``knowledge/``.  Used to check that no label names a
#: document that does not exist -- a typo in a label silently destroys recall.
CORPUS_DOCUMENT_IDS: Final[frozenset[str]] = frozenset({
    "dpi-network-security-terms",
    "dns-normal-behaviour",
    "cdn-and-multi-host-traffic",
    "suspicious-dns-indicators",
    "triaging-unknown-application-traffic",
    "dns-tunneling",
})

PCAP: Final[Path] = Path("test_dpi.pcap")


# ===========================================================================
# Flow builders
# ===========================================================================
def flow(flow_id: int, **overrides) -> FlowRecord:
    """A completed HTTPS flow, with any field overridable."""
    defaults = dict(
        flow_id=flow_id, protocol=TransportProtocol.TCP, dst_port=443,
        src_port=50000 + flow_id, server_name="www.example.com", application="HTTPS",
        state="CLASSIFIED", verdict="FORWARD", packets_out=8, packets_in=12,
        bytes_out=900, bytes_in=6400, syn_seen=True, syn_ack_seen=True,
        fin_seen=True, src_ip="host-1", dst_ip="net-1",
    )
    defaults.update(overrides)
    return FlowRecord(**defaults)


def dns_flow(flow_id: int, name: str, **overrides) -> FlowRecord:
    """A DNS query over UDP/53."""
    defaults = dict(
        protocol=TransportProtocol.UDP, dst_port=53, application="DNS",
        server_name=name, syn_seen=False, syn_ack_seen=False, fin_seen=False,
        packets_out=1, packets_in=1, bytes_out=80, bytes_in=180,
    )
    defaults.update(overrides)
    return flow(flow_id, **defaults)


def build_report(flows: list[FlowRecord], capture_name: str) -> CaptureReport:
    """Assemble a report whose totals are derived from its flows.

    Counters are summed rather than supplied, so a synthetic capture cannot
    claim traffic its flows do not contain.
    """
    applications: dict[str, int] = {}
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
            dropped_packets=0,
            total_flows=len(flows), flows_included=len(flows),
        ),
        application_distribution=dict(sorted(applications.items(),
                                             key=lambda kv: (-kv[1], kv[0]))),
        top_server_names=sorted({f.server_name for f in flows if f.server_name})[:5],
        blocking_rules_active={},
        flows=flows,
        redaction_mode="redact_private",
        notes=[],
    )


# ===========================================================================
# Case captures
# ===========================================================================
#: Twenty hostnames for the browsing baseline.  Ordinary, repeated across
#: flows, and unrelated to any DNS-encoding pattern.
_CDN_HOSTS: Final[tuple[str, ...]] = (
    "www.example.com", "cdn.example.com", "static.example.net", "img.example.net",
    "api.example.org", "fonts.example.com", "assets.example.net", "www.example.org",
)

#: A long, high-entropy leftmost label -- what an encoder produces.
_ENCODED = "k7f2q9x4m1z8b3v6n5c0"


def _dns_high_cardinality() -> CaptureReport:
    """Many DNS flows, each a different name; no encoding-shaped labels."""
    flows = [dns_flow(i, f"host{i:02d}.lookup.example") for i in range(9)]
    flows += [flow(9 + i, server_name=_CDN_HOSTS[i % len(_CDN_HOSTS)]) for i in range(6)]
    return build_report(flows, "dns-high-cardinality.pcap")


def _dns_tunneling() -> CaptureReport:
    """High DNS volume, near-unique encoded names, upload-heavy queries."""
    flows = [
        dns_flow(i, f"{_ENCODED}{i:02d}.tunnel.example",
                 bytes_out=6000, bytes_in=120, packets_out=12, packets_in=2)
        for i in range(18)
    ]
    flows += [flow(18 + i, server_name=_CDN_HOSTS[i % len(_CDN_HOSTS)]) for i in range(4)]
    return build_report(flows, "dns-tunneling.pcap")


def _normal_cdn() -> CaptureReport:
    """Ordinary browsing: many hosts, download-dominant, completed handshakes."""
    flows = [flow(i, server_name=_CDN_HOSTS[i % len(_CDN_HOSTS)]) for i in range(14)]
    flows += [flow(14 + i, protocol=TransportProtocol.UDP, dst_port=443,
                   application="QUIC", server_name=_CDN_HOSTS[i % len(_CDN_HOSTS)],
                   syn_seen=False, syn_ack_seen=False, fin_seen=False) for i in range(4)]
    flows += [dns_flow(18 + i, _CDN_HOSTS[i % len(_CDN_HOSTS)]) for i in range(3)]
    return build_report(flows, "normal-cdn.pcap")


def _suspicious_dns_labels() -> CaptureReport:
    """Encoded-looking labels, but too few flows to trip the volume heuristics."""
    flows = [dns_flow(i, f"{_ENCODED}{i}.lookup.example") for i in range(4)]
    flows += [flow(4 + i, server_name=_CDN_HOSTS[i % len(_CDN_HOSTS)]) for i in range(11)]
    return build_report(flows, "suspicious-dns-labels.pcap")


def _unknown_application() -> CaptureReport:
    """Mostly unclassified TLS with no server name -- a classifier outcome."""
    flows = [flow(i, application="Unknown", server_name=None) for i in range(9)]
    flows += [flow(9 + i, server_name=_CDN_HOSTS[i % len(_CDN_HOSTS)]) for i in range(5)]
    return build_report(flows, "unknown-application.pcap")


def _two_documents() -> CaptureReport:
    """Two unrelated observable patterns at once: encoded DNS and unknown TLS."""
    flows = [dns_flow(i, f"{_ENCODED}{i}.tunnel.example") for i in range(5)]
    flows += [flow(5 + i, application="Unknown", server_name=None) for i in range(8)]
    flows += [flow(13 + i, server_name=_CDN_HOSTS[i % len(_CDN_HOSTS)]) for i in range(3)]
    return build_report(flows, "two-documents.pcap")


def _knowledge_conflict() -> CaptureReport:
    """Plain HTTPS, no DNS at all.

    The conflict case: DNS documents may still be retrieved -- the capture-wide
    query mentions traffic in general -- and the model must not conclude that
    DNS traffic was observed. There is none to observe.
    """
    flows = [flow(i, server_name=_CDN_HOSTS[i % len(_CDN_HOSTS)]) for i in range(6)]
    return build_report(flows, "knowledge-conflict.pcap")


def _real_capture() -> CaptureReport | None:
    """The real ``test_dpi.pcap``, through the unmodified DPI engine."""
    if not PCAP.is_file():
        return None
    from ai.config import AIConfig
    from ai.extractor import build_capture_report
    from dpi.dpi_engine import Config, DPIEngine

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        engine = DPIEngine(Config())
        engine.initialize()
        engine.process_file(str(PCAP), os.devnull)
        snapshot = engine.get_flow_snapshot()
    return build_capture_report(snapshot, str(PCAP), AIConfig())


def pcap_available() -> bool:
    return PCAP.is_file()


# ===========================================================================
# The dataset
# ===========================================================================
@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One capture and everything a correct system should do with it."""

    case_id: str
    group: str = field(metadata={"doc": "A-H, matching the evaluation plan."})
    description: str = ""
    source: str = "synthetic"
    build: Callable[[], CaptureReport | None] = _normal_cdn

    #: Signals the deterministic extractor must produce.
    expected_signals: frozenset[str] = frozenset()
    #: Signals whose presence would be a false detection for this capture.
    forbidden_signals: frozenset[str] = frozenset()

    #: Documents a correct retrieval should surface.
    relevant_documents: frozenset[str] = frozenset()
    #: ``(document_id, section)`` pairs, where a specific section is the answer.
    relevant_sections: frozenset[tuple[str, str]] = frozenset()
    #: Documents whose appearance in the top results would be an error.
    irrelevant_documents: frozenset[str] = frozenset()

    #: Include in the small live-LLM subset.
    live: bool = False
    #: Highest risk level the analysis may report without over-escalating.
    max_risk: str | None = None
    #: Terms that must not appear in ``observed_facts`` for this capture.
    forbidden_fact_terms: tuple[str, ...] = ()

    def report(self) -> CaptureReport | None:
        return self.build()


CASES: Final[tuple[EvaluationCase, ...]] = (
    EvaluationCase(
        case_id="dns-high-cardinality",
        group="A",
        description="Nine DNS queries, every name different, no encoding shape.",
        source="synthetic",
        build=_dns_high_cardinality,
        expected_signals=frozenset({"dns_high_volume", "dns_high_cardinality",
                                    "baseline_web_browsing"}),
        forbidden_signals=frozenset({"dns_anomalous_label", "scan_port_fanout",
                                     "blocked_traffic_present"}),
        relevant_documents=frozenset({"suspicious-dns-indicators", "dns-normal-behaviour",
                                      "dns-tunneling"}),
        relevant_sections=frozenset({
            ("suspicious-dns-indicators", "Indicators"),
            ("dns-normal-behaviour", "Indicators"),
            ("dns-tunneling", "Indicators"),
        }),
        irrelevant_documents=frozenset({"triaging-unknown-application-traffic"}),
        live=True,
        max_risk="medium",
    ),
    EvaluationCase(
        case_id="dns-tunneling",
        group="B",
        description="Eighteen upload-heavy DNS queries with long encoded labels.",
        source="synthetic",
        build=_dns_tunneling,
        expected_signals=frozenset({"dns_high_volume", "dns_high_cardinality",
                                    "dns_anomalous_label", "upload_asymmetry",
                                    "baseline_web_browsing"}),
        forbidden_signals=frozenset({"blocked_traffic_present", "plaintext_http"}),
        relevant_documents=frozenset({"dns-tunneling", "suspicious-dns-indicators",
                                      "dns-normal-behaviour"}),
        relevant_sections=frozenset({
            ("dns-tunneling", "Indicators"),
            ("dns-tunneling", "Summary"),
            ("suspicious-dns-indicators", "Indicators"),
        }),
        irrelevant_documents=frozenset({"triaging-unknown-application-traffic"}),
        live=True,
        max_risk="high",
    ),
    EvaluationCase(
        case_id="normal-cdn-multi-host",
        group="C",
        description="Ordinary browsing across eight hosts, TLS and QUIC, download-heavy.",
        source="synthetic",
        build=_normal_cdn,
        expected_signals=frozenset({"baseline_web_browsing", "quic_present"}),
        forbidden_signals=frozenset({"dns_anomalous_label", "scan_port_fanout",
                                     "scan_half_open", "upload_asymmetry",
                                     "unknown_app_share", "blocked_traffic_present"}),
        relevant_documents=frozenset({"cdn-and-multi-host-traffic",
                                      "dns-normal-behaviour"}),
        relevant_sections=frozenset({
            ("cdn-and-multi-host-traffic", "Summary"),
            ("cdn-and-multi-host-traffic", "Indicators"),
        }),
        irrelevant_documents=frozenset({"dns-tunneling"}),
        live=True,
        max_risk="low",
    ),
    EvaluationCase(
        case_id="suspicious-dns-indicators",
        group="D",
        description="Four encoded-looking DNS names -- shape without volume.",
        source="synthetic",
        build=_suspicious_dns_labels,
        expected_signals=frozenset({"dns_anomalous_label", "baseline_web_browsing"}),
        forbidden_signals=frozenset({"dns_high_volume", "scan_port_fanout"}),
        relevant_documents=frozenset({"suspicious-dns-indicators", "dns-tunneling"}),
        relevant_sections=frozenset({
            ("suspicious-dns-indicators", "Indicators"),
            ("dns-tunneling", "Indicators"),
        }),
        irrelevant_documents=frozenset({"triaging-unknown-application-traffic"}),
        live=False,
        max_risk="medium",
    ),
    EvaluationCase(
        case_id="unknown-application",
        group="E",
        description="Nine unclassified TLS flows with no server name.",
        source="synthetic",
        build=_unknown_application,
        expected_signals=frozenset({"unknown_app_share", "tls_without_sni",
                                    "baseline_web_browsing"}),
        forbidden_signals=frozenset({"dns_high_volume", "dns_anomalous_label",
                                     "blocked_traffic_present"}),
        relevant_documents=frozenset({"triaging-unknown-application-traffic",
                                      "dpi-network-security-terms"}),
        relevant_sections=frozenset({
            ("triaging-unknown-application-traffic", "Summary"),
            ("triaging-unknown-application-traffic", "Recommended checks"),
        }),
        irrelevant_documents=frozenset({"dns-tunneling"}),
        live=True,
        max_risk="low",
    ),
    EvaluationCase(
        case_id="real-capture-benign",
        group="F",
        description="The real test_dpi.pcap: 27 flows of ordinary mixed traffic.",
        source="pcap",
        build=_real_capture,
        expected_signals=frozenset({"plaintext_http", "tls_without_sni",
                                    "baseline_web_browsing"}),
        forbidden_signals=frozenset({"dns_anomalous_label", "scan_port_fanout",
                                     "upload_asymmetry", "blocked_traffic_present"}),
        relevant_documents=frozenset({"cdn-and-multi-host-traffic",
                                      "triaging-unknown-application-traffic",
                                      "dpi-network-security-terms"}),
        relevant_sections=frozenset({
            ("cdn-and-multi-host-traffic", "Summary"),
            ("triaging-unknown-application-traffic", "Summary"),
        }),
        irrelevant_documents=frozenset({"dns-tunneling"}),
        live=True,
        max_risk="low",
    ),
    EvaluationCase(
        case_id="two-documents-relevant",
        group="G",
        description="Encoded DNS names and unclassified TLS in one capture.",
        source="synthetic",
        build=_two_documents,
        expected_signals=frozenset({"dns_anomalous_label", "unknown_app_share",
                                    "tls_without_sni", "baseline_web_browsing"}),
        forbidden_signals=frozenset({"blocked_traffic_present", "plaintext_http"}),
        relevant_documents=frozenset({"dns-tunneling",
                                      "triaging-unknown-application-traffic"}),
        relevant_sections=frozenset({
            ("dns-tunneling", "Indicators"),
            ("triaging-unknown-application-traffic", "Summary"),
        }),
        irrelevant_documents=frozenset(),
        live=False,
        max_risk="medium",
    ),
    EvaluationCase(
        case_id="knowledge-conflicts-observation",
        group="H",
        description="Six plain HTTPS flows and no DNS whatsoever.",
        source="synthetic",
        build=_knowledge_conflict,
        expected_signals=frozenset({"baseline_web_browsing"}),
        forbidden_signals=frozenset({"dns_high_volume", "dns_high_cardinality",
                                     "dns_anomalous_label", "unknown_app_share",
                                     "scan_port_fanout", "blocked_traffic_present"}),
        relevant_documents=frozenset({"cdn-and-multi-host-traffic"}),
        relevant_sections=frozenset({("cdn-and-multi-host-traffic", "Summary")}),
        irrelevant_documents=frozenset({"dns-tunneling", "suspicious-dns-indicators"}),
        live=True,
        max_risk="low",
        # The capture contains no DNS.  If the analysis states one as an
        # observed fact, reference knowledge has manufactured an observation.
        forbidden_fact_terms=("dns", "port 53", "tunnel", "udp"),
    ),
)


def cases_for_live() -> tuple[EvaluationCase, ...]:
    """The small subset sent to a real provider.  Deliberately not all of them."""
    return tuple(case for case in CASES if case.live)
