"""Ethernet / IPv4 / TCP / UDP packet parser.

Python port of ``include/packet_parser.h`` + ``src/packet_parser.cpp``
(C++ ``namespace PacketAnalyzer``).

Decodes a :class:`~dpi.pcap_reader.RawPacket` into a :class:`ParsedPacket`,
walking the layers in order and maintaining a running ``offset`` that ends up
pointing at the payload.  Header lengths are variable: IPv4 via the IHL nibble,
TCP via the data-offset nibble, UDP fixed at 8 bytes.

C++ concepts replaced
---------------------
``*reinterpret_cast<const uint16_t*>(data + 12)`` then ``ntohs``
    An unaligned native-order load followed by a conditional byte swap is,
    on every host, exactly a big-endian load.  So these become
    ``struct.unpack_from(">H", ...)``, which is host-independent *and*
    sidesteps the strict-aliasing/unaligned-access undefined behaviour the
    original relies on.  (The C++ works in practice because x86 tolerates
    unaligned loads; it would fault on stricter architectures.)

``std::memcpy(&src_ip, ip_data + 12, 4)`` for IP addresses
    This is deliberately *not* a big-endian load — it is a **native**-order
    load of the four wire bytes, and ``ipToString`` then shifts by 0/8/16/24.
    On a little-endian host those two quirks cancel and the dotted quad comes
    out correct.  The port reproduces this exactly (``"=I"`` plus the same
    shifts) rather than "fixing" it to a big-endian read, so behaviour matches
    byte for byte.  See :func:`ip_to_string`.

``const uint8_t* payload_data`` into the packet buffer
    Becomes a zero-copy :class:`memoryview` slice over ``raw.data``.

``bool parse(const RawPacket&, ParsedPacket& out)`` (out-parameter)
    Becomes ``parse(raw) -> ParsedPacket | None``.  ``None`` corresponds to the
    ``false`` return.  A mutating ``parse_into`` overload is also provided for
    call sites that reused one ``ParsedPacket`` across a loop.

``static`` member functions
    Become ``@staticmethod``s on :class:`PacketParser`, plus module-level
    aliases so either calling style works.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Final

from .pcap_reader import RawPacket

__all__ = [
    "TCPFlags",
    "Protocol",
    "EtherType",
    "ETH_HEADER_LEN",
    "MIN_IP_HEADER_LEN",
    "MIN_TCP_HEADER_LEN",
    "UDP_HEADER_LEN",
    "EthernetHeader",
    "IPv4Header",
    "TCPHeader",
    "UDPHeader",
    "ParsedPacket",
    "PacketParser",
    "parse",
    "mac_to_string",
    "ip_to_string",
    "protocol_to_string",
    "tcp_flags_to_string",
]


# ============================================================================
# Constants (C++ ``namespace TCPFlags`` / ``Protocol`` / ``EtherType``)
# ============================================================================
class TCPFlags:
    """TCP flag bit masks.  Mirrors ``namespace TCPFlags``."""

    FIN: Final[int] = 0x01
    SYN: Final[int] = 0x02
    RST: Final[int] = 0x04
    PSH: Final[int] = 0x08
    ACK: Final[int] = 0x10
    URG: Final[int] = 0x20


class Protocol:
    """IP protocol numbers.  Mirrors ``namespace Protocol``."""

    ICMP: Final[int] = 1
    TCP: Final[int] = 6
    UDP: Final[int] = 17


class EtherType:
    """EtherType values.  Mirrors ``namespace EtherType``."""

    IPv4: Final[int] = 0x0800
    IPv6: Final[int] = 0x86DD
    ARP: Final[int] = 0x0806


#: Ethernet header is a fixed 14 bytes.
ETH_HEADER_LEN: Final[int] = 14
#: Minimum IPv4 header (IHL == 5).
MIN_IP_HEADER_LEN: Final[int] = 20
#: Minimum TCP header (data offset == 5).
MIN_TCP_HEADER_LEN: Final[int] = 20
#: UDP header is always 8 bytes.
UDP_HEADER_LEN: Final[int] = 8


# ============================================================================
# Wire-format header structs
# ============================================================================
# NOTE: In the C++ original these four structs are declared in the header but
# never actually instantiated -- parsing is done with raw offsets into the
# packet buffer.  They are ported anyway, with working decoders, so nothing
# from the original interface is missing and callers that want a structured
# view of a header can get one.
# ============================================================================
@dataclass(slots=True)
class EthernetHeader:
    """Ethernet header (14 bytes) — the "envelope" for the packet."""

    dest_mac: bytes = b"\x00" * 6  # Destination MAC address
    src_mac: bytes = b"\x00" * 6  # Source MAC address
    ether_type: int = 0  # Type of payload (0x0800 = IPv4)

    @classmethod
    def unpack_from(cls, data: bytes, offset: int = 0) -> "EthernetHeader":
        dest_mac, src_mac, ether_type = struct.unpack_from(">6s6sH", data, offset)
        return cls(dest_mac=dest_mac, src_mac=src_mac, ether_type=ether_type)


@dataclass(slots=True)
class IPv4Header:
    """IPv4 header (20-60 bytes, usually 20)."""

    version_ihl: int = 0  # Version (4 bits) + Header Length (4 bits)
    tos: int = 0  # Type of Service
    total_length: int = 0  # Total packet length
    identification: int = 0  # Fragment identification
    flags_fragment: int = 0  # Flags (3 bits) + Fragment Offset (13 bits)
    ttl: int = 0  # Time To Live
    protocol: int = 0  # Protocol (6=TCP, 17=UDP, 1=ICMP)
    checksum: int = 0  # Header checksum
    src_ip: int = 0  # Source IP address (native-order load, see ip_to_string)
    dest_ip: int = 0  # Destination IP address

    @property
    def version(self) -> int:
        """IP version — the high nibble of ``version_ihl``."""
        return (self.version_ihl >> 4) & 0x0F

    @property
    def ihl(self) -> int:
        """Header length in 32-bit words — the low nibble of ``version_ihl``."""
        return self.version_ihl & 0x0F

    @property
    def header_length(self) -> int:
        """Header length in bytes."""
        return self.ihl * 4

    @classmethod
    def unpack_from(cls, data: bytes, offset: int = 0) -> "IPv4Header":
        (
            version_ihl,
            tos,
            total_length,
            identification,
            flags_fragment,
            ttl,
            protocol,
            checksum,
        ) = struct.unpack_from(">BBHHHBBH", data, offset)
        # Addresses use a NATIVE load, matching the C++ memcpy.
        (src_ip,) = struct.unpack_from("=I", data, offset + 12)
        (dest_ip,) = struct.unpack_from("=I", data, offset + 16)
        return cls(
            version_ihl=version_ihl,
            tos=tos,
            total_length=total_length,
            identification=identification,
            flags_fragment=flags_fragment,
            ttl=ttl,
            protocol=protocol,
            checksum=checksum,
            src_ip=src_ip,
            dest_ip=dest_ip,
        )


@dataclass(slots=True)
class TCPHeader:
    """TCP header (20-60 bytes, usually 20)."""

    src_port: int = 0
    dest_port: int = 0
    seq_number: int = 0
    ack_number: int = 0
    data_offset: int = 0  # Data offset (4 bits) + Reserved (4 bits)
    flags: int = 0  # TCP flags (SYN, ACK, FIN, etc.)
    window: int = 0
    checksum: int = 0
    urgent_pointer: int = 0

    @property
    def header_length(self) -> int:
        """Header length in bytes, from the high nibble of ``data_offset``."""
        return ((self.data_offset >> 4) & 0x0F) * 4

    @classmethod
    def unpack_from(cls, data: bytes, offset: int = 0) -> "TCPHeader":
        fields = struct.unpack_from(">HHIIBBHHH", data, offset)
        return cls(*fields)


@dataclass(slots=True)
class UDPHeader:
    """UDP header (8 bytes - always fixed size)."""

    src_port: int = 0
    dest_port: int = 0
    length: int = 0  # Length of UDP header + data
    checksum: int = 0

    @classmethod
    def unpack_from(cls, data: bytes, offset: int = 0) -> "UDPHeader":
        return cls(*struct.unpack_from(">HHHH", data, offset))


# ============================================================================
# Parsed packet information - human-readable format
# ============================================================================
@dataclass(slots=True)
class ParsedPacket:
    """Decoded view of one packet.

    Mirrors ``struct ParsedPacket``.  Every field defaults to zero/empty, which
    matches the C++ ``parsed = ParsedPacket();`` reset at the top of ``parse``:
    that is value-initialisation of a class with a non-user-provided default
    constructor, so the POD members are zero-initialised.

    Note that ``src_ip``/``dest_ip`` here are **dotted-quad strings**, unlike
    :class:`~dpi.types.FiveTuple`, which keeps them as wire-order integers.
    """

    # Timestamps
    timestamp_sec: int = 0
    timestamp_usec: int = 0

    # Ethernet layer
    src_mac: str = ""
    dest_mac: str = ""
    ether_type: int = 0

    # IP layer (if present)
    has_ip: bool = False
    ip_version: int = 0
    src_ip: str = ""
    dest_ip: str = ""
    protocol: int = 0  # TCP=6, UDP=17, ICMP=1
    ttl: int = 0

    # Transport layer (if present)
    has_tcp: bool = False
    has_udp: bool = False
    src_port: int = 0
    dest_port: int = 0

    # TCP-specific
    tcp_flags: int = 0
    seq_number: int = 0
    ack_number: int = 0

    # Payload
    payload_length: int = 0
    payload_data: memoryview | None = None  # Points into original packet

    def reset(self) -> None:
        """Restore every field to its default, as ``parsed = ParsedPacket()`` did."""
        self.timestamp_sec = 0
        self.timestamp_usec = 0
        self.src_mac = ""
        self.dest_mac = ""
        self.ether_type = 0
        self.has_ip = False
        self.ip_version = 0
        self.src_ip = ""
        self.dest_ip = ""
        self.protocol = 0
        self.ttl = 0
        self.has_tcp = False
        self.has_udp = False
        self.src_port = 0
        self.dest_port = 0
        self.tcp_flags = 0
        self.seq_number = 0
        self.ack_number = 0
        self.payload_length = 0
        self.payload_data = None


# ============================================================================
# Stateless helpers (C++ static members / free helpers)
# ============================================================================
def mac_to_string(mac: bytes | memoryview) -> str:
    """Format 6 MAC bytes as lowercase colon-separated hex.

    Mirrors ``PacketParser::macToString``: ``std::hex`` with
    ``setfill('0') << setw(2)`` produces lowercase, zero-padded pairs.
    """
    return ":".join(f"{b:02x}" for b in bytes(mac[:6]))


def ip_to_string(ip: int) -> str:
    """Format a natively-loaded IPv4 word as a dotted quad.

    Mirrors ``PacketParser::ipToString``, shifting by 0/8/16/24.

    The C++ comment claims the value is "in network byte order (big-endian)",
    but it was produced by ``memcpy`` into a native ``uint32_t`` — a *native*
    load.  On a little-endian host the low byte therefore holds the first wire
    octet and this shift order yields the correct address; on a big-endian host
    the same code would print the octets reversed.  That host-dependence is
    part of the original behaviour and is preserved here verbatim.
    """
    return (
        f"{(ip >> 0) & 0xFF}."
        f"{(ip >> 8) & 0xFF}."
        f"{(ip >> 16) & 0xFF}."
        f"{(ip >> 24) & 0xFF}"
    )


def protocol_to_string(protocol: int) -> str:
    """Name an IP protocol number.  Mirrors ``PacketParser::protocolToString``."""
    if protocol == Protocol.ICMP:
        return "ICMP"
    if protocol == Protocol.TCP:
        return "TCP"
    if protocol == Protocol.UDP:
        return "UDP"
    return f"Unknown({protocol})"


def tcp_flags_to_string(flags: int) -> str:
    """Render TCP flags as a space-separated list.

    Mirrors ``PacketParser::tcpFlagsToString`` exactly, including the fixed
    emission order (SYN, ACK, FIN, RST, PSH, URG — *not* bit order), the
    trailing-space trim, and the ``"none"`` result for no flags set.
    """
    parts: list[str] = []
    if flags & TCPFlags.SYN:
        parts.append("SYN")
    if flags & TCPFlags.ACK:
        parts.append("ACK")
    if flags & TCPFlags.FIN:
        parts.append("FIN")
    if flags & TCPFlags.RST:
        parts.append("RST")
    if flags & TCPFlags.PSH:
        parts.append("PSH")
    if flags & TCPFlags.URG:
        parts.append("URG")
    return " ".join(parts) if parts else "none"


# ============================================================================
# Class to parse raw packets
# ============================================================================
class PacketParser:
    """Stateless packet decoder.  Mirrors ``class PacketParser``.

    All members are static in the C++ original and remain so here.
    """

    __slots__ = ()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    @staticmethod
    def parse(raw: RawPacket) -> ParsedPacket | None:
        """Parse a raw packet, or return ``None`` if it cannot be decoded.

        Mirrors ``bool PacketParser::parse(const RawPacket&, ParsedPacket&)``;
        ``None`` corresponds to the ``false`` return.
        """
        parsed = ParsedPacket()
        return parsed if PacketParser.parse_into(raw, parsed) else None

    @staticmethod
    def parse_into(raw: RawPacket, parsed: ParsedPacket) -> bool:
        """Parse into an existing :class:`ParsedPacket`, returning success.

        This is the literal shape of the C++ signature, for call sites that
        reused a single ``ParsedPacket`` across a read loop.

        Note that on failure ``parsed`` may be left **partially populated** —
        the C++ behaves the same way, because it writes fields before some of
        its length checks.  Callers must not read ``parsed`` after a ``False``.
        """
        # Initialize parsed packet
        parsed.reset()
        parsed.timestamp_sec = raw.header.ts_sec
        parsed.timestamp_usec = raw.header.ts_usec

        data = raw.data
        length = len(data)
        offset = 0

        # Parse Ethernet header first
        ok, offset = PacketParser._parse_ethernet(data, length, parsed, offset)
        if not ok:
            return False

        # Parse IP layer if it's an IPv4 packet
        if parsed.ether_type == EtherType.IPv4:
            ok, offset = PacketParser._parse_ipv4(data, length, parsed, offset)
            if not ok:
                return False

            # Parse transport layer based on protocol
            if parsed.protocol == Protocol.TCP:
                ok, offset = PacketParser._parse_tcp(data, length, parsed, offset)
                if not ok:
                    return False
            elif parsed.protocol == Protocol.UDP:
                ok, offset = PacketParser._parse_udp(data, length, parsed, offset)
                if not ok:
                    return False

        # Set payload information.
        # NOTE: for a non-IPv4 frame (ARP, IPv6, ...) offset is still 14, so
        # everything after the Ethernet header counts as "payload".  Faithful
        # to the original.
        if offset < length:
            parsed.payload_length = length - offset
            parsed.payload_data = memoryview(data)[offset:]
        else:
            parsed.payload_length = 0
            parsed.payload_data = None

        return True

    # ------------------------------------------------------------------
    # Layer parsers.  Each returns (success, new_offset), standing in for the
    # C++ `bool` return plus `size_t& offset` in/out parameter.
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_ethernet(
        data: bytes, length: int, parsed: ParsedPacket, offset: int
    ) -> tuple[bool, int]:
        """Decode the 14-byte Ethernet header.  Mirrors ``parseEthernet``."""
        if length < ETH_HEADER_LEN:
            return False, offset  # Packet too short

        # Parse destination MAC (bytes 0-5)
        parsed.dest_mac = mac_to_string(data[0:6])

        # Parse source MAC (bytes 6-11)
        parsed.src_mac = mac_to_string(data[6:12])

        # Parse EtherType (bytes 12-13, big-endian)
        (parsed.ether_type,) = struct.unpack_from(">H", data, 12)

        return True, ETH_HEADER_LEN

    @staticmethod
    def _parse_ipv4(
        data: bytes, length: int, parsed: ParsedPacket, offset: int
    ) -> tuple[bool, int]:
        """Decode the variable-length IPv4 header.  Mirrors ``parseIPv4``."""
        if length < offset + MIN_IP_HEADER_LEN:
            return False, offset  # Packet too short

        # First byte: version (4 bits) + IHL (4 bits)
        version_ihl = data[offset]
        parsed.ip_version = (version_ihl >> 4) & 0x0F
        ihl = version_ihl & 0x0F  # Header length in 32-bit words

        if parsed.ip_version != 4:
            return False, offset  # Not IPv4

        ip_header_len = ihl * 4  # Convert to bytes
        if ip_header_len < MIN_IP_HEADER_LEN or length < offset + ip_header_len:
            return False, offset

        # Parse fields
        parsed.ttl = data[offset + 8]
        parsed.protocol = data[offset + 9]

        # Source IP (bytes 12-15) and Destination IP (bytes 16-19).
        # NATIVE-order loads, matching the C++ memcpy -- see ip_to_string().
        (src_ip,) = struct.unpack_from("=I", data, offset + 12)
        (dest_ip,) = struct.unpack_from("=I", data, offset + 16)
        parsed.src_ip = ip_to_string(src_ip)
        parsed.dest_ip = ip_to_string(dest_ip)

        parsed.has_ip = True
        return True, offset + ip_header_len

    @staticmethod
    def _parse_tcp(
        data: bytes, length: int, parsed: ParsedPacket, offset: int
    ) -> tuple[bool, int]:
        """Decode the variable-length TCP header.  Mirrors ``parseTCP``."""
        if length < offset + MIN_TCP_HEADER_LEN:
            return False, offset

        # Ports, sequence and ack numbers (big-endian)
        parsed.src_port, parsed.dest_port = struct.unpack_from(">HH", data, offset)
        parsed.seq_number, parsed.ack_number = struct.unpack_from(">II", data, offset + 4)

        # Data offset (upper 4 bits of byte 12) - header length in 32-bit words
        data_offset = (data[offset + 12] >> 4) & 0x0F
        tcp_header_len = data_offset * 4

        # Flags (byte 13)
        parsed.tcp_flags = data[offset + 13]

        # NOTE: the original assigns the fields above BEFORE this validation,
        # so a malformed TCP header leaves them written even though the parse
        # fails.  Preserved.
        if tcp_header_len < MIN_TCP_HEADER_LEN or length < offset + tcp_header_len:
            return False, offset

        parsed.has_tcp = True
        return True, offset + tcp_header_len

    @staticmethod
    def _parse_udp(
        data: bytes, length: int, parsed: ParsedPacket, offset: int
    ) -> tuple[bool, int]:
        """Decode the fixed 8-byte UDP header.  Mirrors ``parseUDP``."""
        if length < offset + UDP_HEADER_LEN:
            return False, offset

        # Source port (bytes 0-1) and destination port (bytes 2-3)
        parsed.src_port, parsed.dest_port = struct.unpack_from(">HH", data, offset)

        parsed.has_udp = True
        return True, offset + UDP_HEADER_LEN

    # ------------------------------------------------------------------
    # Helper functions to convert to human-readable strings
    # ------------------------------------------------------------------
    mac_to_string = staticmethod(mac_to_string)
    ip_to_string = staticmethod(ip_to_string)
    protocol_to_string = staticmethod(protocol_to_string)
    tcp_flags_to_string = staticmethod(tcp_flags_to_string)

    # C++-style aliases
    macToString = staticmethod(mac_to_string)
    ipToString = staticmethod(ip_to_string)
    protocolToString = staticmethod(protocol_to_string)
    tcpFlagsToString = staticmethod(tcp_flags_to_string)


#: Module-level convenience alias for ``PacketParser.parse``.
parse = PacketParser.parse


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys

    from .pcap_reader import PcapReader

    if len(sys.argv) < 2:
        print("Usage: python -m dpi.packet_parser <pcap_file>", file=sys.stderr)
        raise SystemExit(1)

    # Helper string formatting
    assert mac_to_string(bytes([0x00, 0x1A, 0x2B, 0x3C, 0x4D, 0x5E])) == "00:1a:2b:3c:4d:5e"
    assert ip_to_string(0x0A01A8C0) == "192.168.1.10"
    assert protocol_to_string(6) == "TCP"
    assert protocol_to_string(99) == "Unknown(99)"
    assert tcp_flags_to_string(0) == "none"
    assert tcp_flags_to_string(TCPFlags.SYN) == "SYN"
    assert tcp_flags_to_string(TCPFlags.SYN | TCPFlags.ACK) == "SYN ACK"
    # Emission order is SYN ACK FIN RST PSH URG, not numeric bit order:
    assert tcp_flags_to_string(TCPFlags.FIN | TCPFlags.ACK) == "ACK FIN"
    assert tcp_flags_to_string(0x3F) == "SYN ACK FIN RST PSH URG"

    with PcapReader() as reader:
        if not reader.open(sys.argv[1]):
            raise SystemExit(1)

        total = parsed_ok = tcp = udp = other = 0
        for raw in reader:
            total += 1
            pkt = PacketParser.parse(raw)
            if pkt is None:
                continue
            parsed_ok += 1
            if pkt.has_tcp:
                tcp += 1
            elif pkt.has_udp:
                udp += 1
            else:
                other += 1
            if parsed_ok <= 3:
                print(
                    f"  #{total} {pkt.src_ip}:{pkt.src_port} -> "
                    f"{pkt.dest_ip}:{pkt.dest_port} "
                    f"{protocol_to_string(pkt.protocol)} "
                    f"[{tcp_flags_to_string(pkt.tcp_flags)}] "
                    f"payload={pkt.payload_length}"
                )

        print(f"\nTotal: {total}  parsed: {parsed_ok}  TCP: {tcp}  UDP: {udp}  other: {other}")
        print("packet_parser.py self-test OK")
