#!/usr/bin/env python3
"""Working DPI Engine — single-threaded but functional.

Python port of ``src/main_working.cpp``.

This entry point is self-contained: it does **not** use ``RuleManager``,
``ConnectionTracker`` or ``FastPathProcessor``.  It carries its own simplified
``BlockingRules`` and flow table, so its behaviour differs from ``main_dpi`` in
ways worth knowing:

* Domain blocking is a plain **substring** match against the SNI
  (``sni.find(dom) != npos``), with no wildcard support and no case folding.
  Blocking ``"tube"`` here stops YouTube; in ``main_dpi`` it would not.
* There is no port blocking and no ``*.`` pattern handling.
* Classification re-runs on every packet of a flow until an SNI is found,
  rather than latching after the first classified packet.

Because it is single-threaded, its output PCAP is byte-for-byte deterministic —
unlike ``main_dpi``/``dpi_mt``, whose FP threads interleave.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dpi.packet_parser import PacketParser
from dpi.pcap_reader import PcapReader
from dpi.sni_extractor import HTTPHostExtractor, SNIExtractor
from dpi.types import AppType, FiveTuple, app_type_to_string, sni_to_app_type

_PACKET_HEADER_FMT = "=IIII"
_GLOBAL_HEADER_FMT = "=IHHiIII"


@dataclass(slots=True)
class Flow:
    """Simplified connection tracking.  Mirrors ``struct Flow``."""

    tuple: FiveTuple | None = None
    app_type: AppType = AppType.UNKNOWN
    sni: str = ""
    packets: int = 0
    bytes: int = 0
    blocked: bool = False


def _parse_ip(ip: str) -> int:
    """Mirrors the ``parseIP`` lambda (a fourth copy of the same routine)."""
    result = 0
    octet = 0
    shift = 0
    for c in ip:
        if c == ".":
            result |= (octet << shift) & 0xFFFFFFFF
            shift += 8
            octet = 0
        elif "0" <= c <= "9":
            octet = octet * 10 + (ord(c) - 48)
    return (result | ((octet << shift) & 0xFFFFFFFF)) & 0xFFFFFFFF


class BlockingRules:
    """Simplified rule set.  Mirrors ``class BlockingRules``."""

    __slots__ = ("blocked_ips", "blocked_apps", "blocked_domains")

    def __init__(self) -> None:
        self.blocked_ips: set[int] = set()
        self.blocked_apps: set[AppType] = set()
        self.blocked_domains: list[str] = []  # Simple substring match

    def block_ip(self, ip: str) -> None:
        self.blocked_ips.add(_parse_ip(ip))
        print(f"[Rules] Blocked IP: {ip}")

    def block_app(self, app: str) -> None:
        for i in range(int(AppType.APP_COUNT)):
            if app_type_to_string(AppType(i)) == app:
                self.blocked_apps.add(AppType(i))
                print(f"[Rules] Blocked app: {app}")
                return
        print(f"[Rules] Unknown app: {app}", file=sys.stderr)

    def block_domain(self, domain: str) -> None:
        self.blocked_domains.append(domain)
        print(f"[Rules] Blocked domain: {domain}")

    def is_blocked(self, src_ip: int, app: AppType, sni: str) -> bool:
        """Substring domain matching — see the module docstring."""
        if src_ip in self.blocked_ips:
            return True
        if app in self.blocked_apps:
            return True
        return any(dom in sni for dom in self.blocked_domains)


def print_usage(prog: str) -> None:
    """Mirrors ``printUsage``."""
    print(f"""
DPI Engine - Deep Packet Inspection System
==========================================

Usage: {prog} <input.pcap> <output.pcap> [options]

Options:
  --block-ip <ip>        Block traffic from source IP
  --block-app <app>      Block application (YouTube, Facebook, etc.)
  --block-domain <dom>   Block domain (substring match)

Example:
  {prog} capture.pcap filtered.pcap --block-app YouTube --block-ip 192.168.1.50
""", end="")


def main(argv: list[str] | None = None) -> int:
    """Mirrors ``int main(int argc, char* argv[])``."""
    argv = list(sys.argv if argv is None else argv)
    argc = len(argv)

    if argc < 3:
        print_usage(argv[0])
        return 1

    input_file = argv[1]
    output_file = argv[2]

    rules = BlockingRules()

    # Parse options
    i = 3
    while i < argc:
        arg = argv[i]
        if arg == "--block-ip" and i + 1 < argc:
            i += 1
            rules.block_ip(argv[i])
        elif arg == "--block-app" and i + 1 < argc:
            i += 1
            rules.block_app(argv[i])
        elif arg == "--block-domain" and i + 1 < argc:
            i += 1
            rules.block_domain(argv[i])
        i += 1

    print("")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    DPI ENGINE v1.0                            ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Open input
    reader = PcapReader()
    if not reader.open(input_file):
        return 1

    # Open output
    try:
        output = Path(output_file).open("wb")
    except OSError:
        print("Error: Cannot open output file", file=sys.stderr)
        return 1

    # Write PCAP header
    h = reader.get_global_header()
    output.write(
        struct.pack(
            _GLOBAL_HEADER_FMT,
            h.magic_number, h.version_major, h.version_minor,
            h.thiszone, h.sigfigs, h.snaplen, h.network,
        )
    )

    # Flow table
    flows: dict[FiveTuple, Flow] = {}

    # Statistics
    total_packets = 0
    forwarded = 0
    dropped = 0
    app_stats: dict[AppType, int] = {}

    print("[DPI] Processing packets...")

    while True:
        raw = reader.read_next_packet()
        if raw is None:
            break

        total_packets += 1

        parsed = PacketParser.parse(raw)
        if parsed is None:
            continue
        if not parsed.has_ip or (not parsed.has_tcp and not parsed.has_udp):
            continue

        # Create five-tuple
        tuple_ = FiveTuple(
            src_ip=_parse_ip(parsed.src_ip),
            dst_ip=_parse_ip(parsed.dest_ip),
            src_port=parsed.src_port,
            dst_port=parsed.dest_port,
            protocol=parsed.protocol,
        )

        # Get or create flow (C++ operator[] default-constructs)
        flow = flows.get(tuple_)
        if flow is None:
            flow = Flow()
            flows[tuple_] = flow
        if flow.packets == 0:
            flow.tuple = tuple_
        flow.packets += 1
        flow.bytes += len(raw.data)

        # Try SNI extraction - even for flows already marked as generic HTTPS
        if (
            flow.app_type in (AppType.UNKNOWN, AppType.HTTPS)
            and not flow.sni
            and parsed.has_tcp
            and parsed.dest_port == 443
        ):
            payload_offset = 14
            ip_ihl = raw.data[14] & 0x0F
            payload_offset += ip_ihl * 4

            if payload_offset + 12 < len(raw.data):
                tcp_offset = (raw.data[payload_offset + 12] >> 4) & 0x0F
                payload_offset += tcp_offset * 4

                if payload_offset < len(raw.data):
                    payload_len = len(raw.data) - payload_offset
                    if payload_len > 5:  # Minimum TLS record header
                        sni = SNIExtractor.extract(
                            memoryview(raw.data)[payload_offset:], payload_len
                        )
                        if sni is not None:
                            flow.sni = sni
                            flow.app_type = sni_to_app_type(sni)

        # HTTP Host extraction
        if (
            flow.app_type in (AppType.UNKNOWN, AppType.HTTP)
            and not flow.sni
            and parsed.has_tcp
            and parsed.dest_port == 80
        ):
            payload_offset = 14
            ip_ihl = raw.data[14] & 0x0F
            payload_offset += ip_ihl * 4

            if payload_offset + 12 < len(raw.data):
                tcp_offset = (raw.data[payload_offset + 12] >> 4) & 0x0F
                payload_offset += tcp_offset * 4

                if payload_offset < len(raw.data):
                    payload_len = len(raw.data) - payload_offset
                    host = HTTPHostExtractor.extract(
                        memoryview(raw.data)[payload_offset:], payload_len
                    )
                    if host is not None:
                        flow.sni = host
                        flow.app_type = sni_to_app_type(host)

        # DNS classification
        if flow.app_type == AppType.UNKNOWN and (
            parsed.dest_port == 53 or parsed.src_port == 53
        ):
            flow.app_type = AppType.DNS

        # Port-based fallback
        if flow.app_type == AppType.UNKNOWN:
            if parsed.dest_port == 443:
                flow.app_type = AppType.HTTPS
            elif parsed.dest_port == 80:
                flow.app_type = AppType.HTTP

        # Check blocking rules
        if not flow.blocked:
            flow.blocked = rules.is_blocked(tuple_.src_ip, flow.app_type, flow.sni)
            if flow.blocked:
                line = (
                    f"[BLOCKED] {parsed.src_ip} -> {parsed.dest_ip}"
                    f" ({app_type_to_string(flow.app_type)}"
                )
                if flow.sni:
                    line += f": {flow.sni}"
                line += ")"
                print(line)

        # Update app stats
        app_stats[flow.app_type] = app_stats.get(flow.app_type, 0) + 1

        # Forward or drop
        if flow.blocked:
            dropped += 1
        else:
            forwarded += 1
            # Write to output
            size = len(raw.data)
            output.write(
                struct.pack(_PACKET_HEADER_FMT, raw.header.ts_sec, raw.header.ts_usec, size, size)
            )
            output.write(raw.data)

    reader.close()
    output.close()

    # Print report
    print("")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                      PROCESSING REPORT                       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║ Total Packets:      {total_packets:>10}                             ║")
    print(f"║ Forwarded:          {forwarded:>10}                             ║")
    print(f"║ Dropped:            {dropped:>10}                             ║")
    print(f"║ Active Flows:       {len(flows):>10}                             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║                    APPLICATION BREAKDOWN                     ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    # Sort by count (std::sort is unstable in C++; sorted() is stable here)
    sorted_apps = sorted(app_stats.items(), key=lambda kv: -kv[1])

    for app, count in sorted_apps:
        # NOTE: no total_packets == 0 guard in the original -- an empty capture
        # never reaches this loop, so the division is unreachable when zero.
        pct = 100.0 * count / total_packets
        bar = "#" * int(pct / 5)
        print(f"║ {app_type_to_string(app):<15}{count:>8} {pct:>5.1f}% {bar:<20}  ║")

    print("╚══════════════════════════════════════════════════════════════╝")

    # List unique SNIs
    print("\n[Detected Applications/Domains]")
    unique_snis: dict[str, AppType] = {}
    for flow in flows.values():
        if flow.sni:
            unique_snis[flow.sni] = flow.app_type
    for sni, app in unique_snis.items():
        print(f"  - {sni} -> {app_type_to_string(app)}")

    print(f"\nOutput written to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
