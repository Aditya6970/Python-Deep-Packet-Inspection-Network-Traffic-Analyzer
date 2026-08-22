#!/usr/bin/env python3
"""Simple single-threaded test version.

Python port of ``src/main_simple.cpp``.

Note this file recomputes the payload offset by hand instead of using the
offsets ``PacketParser`` already produced — a third copy of that arithmetic in
the project.  Preserved, including its missing bounds check: the C++ indexes
``raw.data[14]`` and ``raw.data[payload_offset + 12]`` without verifying the
packet is long enough.  In C++ those are out-of-bounds reads on a short frame;
Python raises :class:`IndexError` instead, which is the same defect surfaced
safely rather than silently.
"""

from __future__ import annotations

import sys

from dpi.packet_parser import PacketParser
from dpi.pcap_reader import PcapReader
from dpi.sni_extractor import SNIExtractor


def main(argv: list[str] | None = None) -> int:
    """Mirrors ``int main(int argc, char* argv[])``."""
    argv = list(sys.argv if argv is None else argv)

    if len(argv) < 2:
        print(f"Usage: {argv[0]} <pcap_file>", file=sys.stderr)
        return 1

    reader = PcapReader()
    if not reader.open(argv[1]):
        return 1

    count = 0
    tls_count = 0

    print("Processing packets...")

    while True:
        raw = reader.read_next_packet()
        if raw is None:
            break

        count += 1

        parsed = PacketParser.parse(raw)
        if parsed is None:
            continue

        if not parsed.has_ip:
            continue

        line = (
            f"Packet {count}: {parsed.src_ip}:{parsed.src_port}"
            f" -> {parsed.dest_ip}:{parsed.dest_port}"
        )

        # Try SNI extraction for HTTPS packets
        if parsed.has_tcp and parsed.dest_port == 443 and parsed.payload_length > 0:
            # Calculate payload offset
            payload_offset = 14  # Ethernet
            ip_ihl = raw.data[14] & 0x0F
            payload_offset += ip_ihl * 4
            tcp_offset = (raw.data[payload_offset + 12] >> 4) & 0x0F
            payload_offset += tcp_offset * 4

            if payload_offset < len(raw.data):
                payload_len = len(raw.data) - payload_offset
                sni = SNIExtractor.extract(
                    memoryview(raw.data)[payload_offset:], payload_len
                )
                if sni is not None:
                    line += f" [SNI: {sni}]"
                    tls_count += 1

        print(line)

    print(f"\nTotal packets: {count}")
    print(f"SNI extracted: {tls_count}")

    reader.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
