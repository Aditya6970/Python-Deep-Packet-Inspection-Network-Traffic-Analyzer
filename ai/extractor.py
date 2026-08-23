"""Turn a DPI snapshot into a sanitized, validated :class:`~ai.schemas.CaptureReport`.

Pure and deterministic: no network access, no LLM, no clock, no randomness.
The same snapshot always produces the same report, which is what makes this
stage testable offline and useful on its own (a JSON export of flow records).

Two guarantees are structural rather than procedural:

* **No packet payloads.** Records are built field by field from named
  attributes. ``PacketJob.data`` and every ``memoryview`` are never referenced,
  so payload bytes have no path into a report even by accident.
* **No duration.** ``Connection.first_seen``/``last_seen`` are
  :func:`time.monotonic` values measuring *processing* time, not capture time.
  A duration derived from them would be a fabricated number for the model to
  reason from, so it is omitted entirely.
"""

from __future__ import annotations

from pathlib import Path

from dpi.dpi_engine import FlowSnapshot
from dpi.types import AppType, Connection, app_type_to_string

from .config import AIConfig, IPRedactionMode
from .redaction import HostPseudonymiser, redact_ip, sanitize_hostname
from .schemas import (
    CaptureReport,
    CaptureTotals,
    FlowRecord,
    TransportProtocol,
)

__all__ = ["build_capture_report", "flow_sort_key"]

_PROTOCOL_NAMES: dict[int, TransportProtocol] = {
    6: TransportProtocol.TCP,
    17: TransportProtocol.UDP,
}


def flow_sort_key(conn: Connection) -> tuple[int, int, int]:
    """Rank flows by analytical interest, most interesting first.

    When a capture has more flows than :attr:`~ai.config.AIConfig.max_flows`,
    this decides which survive. Blocked flows rank first (they are the ones a
    reader cares about), then classified flows, then by traffic volume.

    Deterministic: no ties are broken by dict or set ordering.
    """
    from dpi.types import ConnectionState

    blocked = 1 if conn.state == ConnectionState.BLOCKED else 0
    classified = 1 if conn.sni else 0
    volume = conn.bytes_out + conn.bytes_in
    return (blocked, classified, volume)


def build_capture_report(
    snapshot: FlowSnapshot,
    capture_path: str | Path,
    config: AIConfig | None = None,
) -> CaptureReport:
    """Build the sanitized report that will be sent for analysis.

    ``capture_path`` contributes only its **file name**; the directory is
    dropped, since a full path can leak a username or directory structure.
    """
    cfg = config or AIConfig()
    pseudonymiser = HostPseudonymiser()
    notes: list[str] = []

    # --- select flows ------------------------------------------------------
    all_connections = list(snapshot.connections)
    total_flows = len(all_connections)

    selected = sorted(all_connections, key=flow_sort_key, reverse=True)
    if total_flows > cfg.max_flows:
        selected = selected[: cfg.max_flows]
        notes.append(
            f"Capture contains {total_flows} flows; the {cfg.max_flows} most "
            f"significant are included (blocked first, then classified, then by volume). "
            f"{total_flows - cfg.max_flows} flows are not shown."
        )

    # Stable ordering for reproducible flow_ids.
    selected.sort(key=lambda c: (c.tuple.dst_port, c.tuple.src_port, c.tuple.src_ip))

    # --- build records -----------------------------------------------------
    flows: list[FlowRecord] = []
    for index, conn in enumerate(selected):
        flows.append(
            FlowRecord(
                flow_id=index,
                protocol=_PROTOCOL_NAMES.get(conn.tuple.protocol, TransportProtocol.OTHER),
                dst_port=conn.tuple.dst_port,
                src_port=conn.tuple.src_port,
                server_name=sanitize_hostname(conn.sni or None),
                application=app_type_to_string(conn.app_type),
                state=conn.state.name,
                verdict=conn.action.name,
                packets_out=conn.packets_out,
                packets_in=conn.packets_in,
                bytes_out=conn.bytes_out,
                bytes_in=conn.bytes_in,
                syn_seen=conn.syn_seen,
                syn_ack_seen=conn.syn_ack_seen,
                fin_seen=conn.fin_seen,
                src_ip=redact_ip(conn.tuple.src_ip, cfg.ip_mode, pseudonymiser),
                dst_ip=redact_ip(conn.tuple.dst_ip, cfg.ip_mode, pseudonymiser),
            )
        )

    # --- capture-wide context ---------------------------------------------
    stats = snapshot.packet_stats
    totals = CaptureTotals(
        total_packets=stats.get("total_packets", 0),
        total_bytes=stats.get("total_bytes", 0),
        tcp_packets=stats.get("tcp_packets", 0),
        udp_packets=stats.get("udp_packets", 0),
        forwarded_packets=stats.get("forwarded_packets", 0),
        dropped_packets=stats.get("dropped_packets", 0),
        total_flows=total_flows,
        flows_included=len(flows),
    )

    app_distribution: dict[str, int] = {}
    for app, count in snapshot.app_distribution.items():
        name = app_type_to_string(app) if isinstance(app, AppType) else str(app)
        app_distribution[name] = app_distribution.get(name, 0) + count
    app_distribution = dict(
        sorted(app_distribution.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    top_names: list[str] = []
    for domain, _count in snapshot.top_domains:
        cleaned = sanitize_hostname(domain)
        if cleaned:
            top_names.append(cleaned)

    rules: dict[str, int] = {}
    if snapshot.rule_stats is not None:
        rules = {
            "blocked_ips": snapshot.rule_stats.blocked_ips,
            "blocked_apps": snapshot.rule_stats.blocked_apps,
            "blocked_domains": snapshot.rule_stats.blocked_domains,
            "blocked_ports": snapshot.rule_stats.blocked_ports,
        }

    if cfg.ip_mode is IPRedactionMode.REDACT_PRIVATE and len(pseudonymiser):
        notes.append(
            f"{len(pseudonymiser)} internal address(es) replaced with stable pseudonyms."
        )
    elif cfg.ip_mode is IPRedactionMode.NONE:
        notes.append("IP addresses were withheld entirely.")

    return CaptureReport(
        capture_name=Path(capture_path).name,
        totals=totals,
        application_distribution=app_distribution,
        top_server_names=top_names,
        blocking_rules_active=rules,
        flows=flows,
        redaction_mode=cfg.ip_mode.value,
        notes=notes,
    )
