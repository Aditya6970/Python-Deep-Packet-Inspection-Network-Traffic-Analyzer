#!/usr/bin/env python3
"""Packet Analyzer CLI — the single-threaded pcap dump tool.

Python port of ``src/main.cpp``.  This is the only target ``CMakeLists.txt``
actually builds.

C++ concepts replaced
---------------------
``std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S")``
    Becomes :meth:`datetime.datetime.fromtimestamp` + ``strftime``.  Both
    render in **local time**, so output depends on the machine's timezone —
    preserved rather than normalised to UTC.

``std::hex``/``std::setw`` stream manipulators
    Become format specs.  Note ``ether_type`` prints as 4 hex digits and
    payload bytes as 2, both lowercase, matching ``std::hex`` defaults.
"""

from __future__ import annotations

import sys
from datetime import datetime

from dpi.packet_parser import EtherType, PacketParser, ParsedPacket
from dpi.pcap_reader import PcapReader


def print_packet_summary(pkt: ParsedPacket, packet_num: int) -> None:
    """Print one packet's decoded layers.  Mirrors ``printPacketSummary``."""
    # Format timestamp (local time, as std::localtime does)
    stamp = datetime.fromtimestamp(pkt.timestamp_sec).strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n========== Packet #{packet_num} ==========")
    print(f"Time: {stamp}.{pkt.timestamp_usec:06d}")

    # Ethernet layer
    print("\n[Ethernet]")
    print(f"  Source MAC:      {pkt.src_mac}")
    print(f"  Destination MAC: {pkt.dest_mac}")

    suffix = ""
    if pkt.ether_type == EtherType.IPv4:
        suffix = " (IPv4)"
    elif pkt.ether_type == EtherType.IPv6:
        suffix = " (IPv6)"
    elif pkt.ether_type == EtherType.ARP:
        suffix = " (ARP)"
    print(f"  EtherType:       0x{pkt.ether_type:04x}{suffix}")

    # IP layer
    if pkt.has_ip:
        print(f"\n[IPv{pkt.ip_version}]")
        print(f"  Source IP:      {pkt.src_ip}")
        print(f"  Destination IP: {pkt.dest_ip}")
        print(f"  Protocol:       {PacketParser.protocol_to_string(pkt.protocol)}")
        print(f"  TTL:            {pkt.ttl}")

    # TCP layer
    if pkt.has_tcp:
        print("\n[TCP]")
        print(f"  Source Port:      {pkt.src_port}")
        print(f"  Destination Port: {pkt.dest_port}")
        print(f"  Sequence Number:  {pkt.seq_number}")
        print(f"  Ack Number:       {pkt.ack_number}")
        print(f"  Flags:            {PacketParser.tcp_flags_to_string(pkt.tcp_flags)}")

    # UDP layer
    if pkt.has_udp:
        print("\n[UDP]")
        print(f"  Source Port:      {pkt.src_port}")
        print(f"  Destination Port: {pkt.dest_port}")

    # Payload info
    if pkt.payload_length > 0:
        print("\n[Payload]")
        print(f"  Length: {pkt.payload_length} bytes")

        # Print first 32 bytes of payload as hex (if present)
        preview_len = min(pkt.payload_length, 32)
        assert pkt.payload_data is not None
        hex_bytes = "".join(f"{pkt.payload_data[i]:02x} " for i in range(preview_len))
        ellipsis = "..." if pkt.payload_length > 32 else ""
        print(f"  Preview: {hex_bytes}{ellipsis}")


def print_usage(program_name: str) -> None:
    """Print the usage banner.  Mirrors ``printUsage``."""
    print(f"Usage: {program_name} <pcap_file> [max_packets]")
    print("\nArguments:")
    print("  pcap_file   - Path to a .pcap file captured by Wireshark")
    print("  max_packets - (Optional) Maximum number of packets to display")
    print("\nExample:")
    print(f"  {program_name} capture.pcap")
    print(f"  {program_name} capture.pcap 10")


def main(argv: list[str] | None = None) -> int:
    """Mirrors ``int main(int argc, char* argv[])``."""
    argv = list(sys.argv if argv is None else argv)
    argc = len(argv)

    print("====================================")
    print("     Packet Analyzer v1.0")
    print("====================================\n")

    # Check command line arguments
    if argc < 2:
        print_usage(argv[0])
        return 1

    filename = argv[1]
    max_packets = -1  # -1 means no limit

    if argc >= 3:
        max_packets = int(argv[2])

    # Open the PCAP file
    reader = PcapReader()
    if not reader.open(filename):
        return 1

    print("\n--- Reading packets ---")

    # Read and parse packets
    packet_count = 0
    parse_errors = 0

    while True:
        raw_packet = reader.read_next_packet()
        if raw_packet is None:
            break

        packet_count += 1

        parsed_packet = PacketParser.parse(raw_packet)
        if parsed_packet is not None:
            print_packet_summary(parsed_packet, packet_count)
        else:
            print(f"Warning: Failed to parse packet #{packet_count}", file=sys.stderr)
            parse_errors += 1

        # Check if we've reached the limit
        if 0 < max_packets <= packet_count:
            print(f"\n(Stopped after {max_packets} packets)")
            break

    # Summary
    print("\n====================================")
    print("Summary:")
    print(f"  Total packets read:  {packet_count}")
    print(f"  Parse errors:        {parse_errors}")
    print("====================================")

    reader.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
