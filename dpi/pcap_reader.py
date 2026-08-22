"""PCAP file reader.

Python port of ``include/pcap_reader.h`` + ``src/pcap_reader.cpp``
(C++ ``namespace PacketAnalyzer``).

Reads classic libpcap-format capture files: a 24-byte global header followed by
a stream of (16-byte packet header, packet bytes) records.  Endianness is
detected from the magic number and applied to every subsequent field.

C++ concepts replaced
---------------------
``file_.read(reinterpret_cast<char*>(&global_header_), sizeof(...))``
    Reading raw bytes straight over a struct is replaced by explicit
    :mod:`struct` unpacking.  The format strings use ``=`` (native byte order,
    standard sizes, **no** alignment padding), which reproduces the C++ layout
    exactly: both structs are naturally aligned with no interior padding, so
    ``sizeof(PcapGlobalHeader) == 24`` and ``sizeof(PcapPacketHeader) == 16``.
    Those sizes are asserted at import.

``std::ifstream`` + RAII destructor
    Becomes a :class:`pathlib.Path` opened in binary mode plus explicit
    :meth:`PcapReader.close`.  Python has no deterministic destructor, so the
    class is a **context manager** — ``with PcapReader() as reader:`` is the
    faithful equivalent of the C++ object going out of scope.

``bool readNextPacket(RawPacket& out)`` (out-parameter + status flag)
    Becomes ``read_next_packet() -> RawPacket | None``.  The C++ loop
    ``while (reader.readNextPacket(raw))`` becomes
    ``while (raw := reader.read_next_packet()) is not None``, or simply
    ``for raw in reader``.

``std::vector<uint8_t> data``
    Becomes ``bytes``.  The C++ code reused one buffer via ``resize()``; Python
    allocates per packet, which is semantically identical here because nothing
    holds a reference to the previous packet's bytes across iterations.

Endianness detection is host-relative, exactly as in the original
--------------------------------------------------------------------
The magic number is compared *after* being read in **native** order, so on a
little-endian host a little-endian capture yields ``needs_byte_swap == False``.
The comparison inverts on a big-endian host — that is the C++ behaviour and it
is preserved rather than normalised.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Final, Iterator

__all__ = [
    "PCAP_MAGIC_NATIVE",
    "PCAP_MAGIC_SWAPPED",
    "PCAP_MAGIC_NATIVE_NS",
    "PCAP_MAGIC_SWAPPED_NS",
    "GLOBAL_HEADER_SIZE",
    "PACKET_HEADER_SIZE",
    "PcapGlobalHeader",
    "PcapPacketHeader",
    "RawPacket",
    "PcapReader",
]

# ---------------------------------------------------------------------------
# Magic numbers for PCAP files
# ---------------------------------------------------------------------------
#: Magic reading as this value in **native** order needs no swapping.
PCAP_MAGIC_NATIVE: Final[int] = 0xA1B2C3D4
#: Magic reading as this value in native order means the file is byte-swapped.
PCAP_MAGIC_SWAPPED: Final[int] = 0xD4C3B2A1
#: Nanosecond-resolution variants of the same two magics (libpcap 1.5+,
#: written by `tcpdump --time-stamp-precision=nano`).  FIXED: the C++ rejected
#: these outright with "Invalid PCAP magic number", so a modern nanosecond
#: capture could not be read at all.
PCAP_MAGIC_NATIVE_NS: Final[int] = 0xA1B23C4D
PCAP_MAGIC_SWAPPED_NS: Final[int] = 0x4D3CB2A1

# ---------------------------------------------------------------------------
# Binary layouts.  '=' selects native byte order with standard sizes and no
# alignment padding, matching the padding-free C++ structs.
# ---------------------------------------------------------------------------
#: magic_number, version_major, version_minor, thiszone, sigfigs, snaplen, network
_GLOBAL_HEADER_FMT: Final[str] = "=IHHiIII"
#: ts_sec, ts_usec, incl_len, orig_len
_PACKET_HEADER_FMT: Final[str] = "=IIII"

GLOBAL_HEADER_SIZE: Final[int] = struct.calcsize(_GLOBAL_HEADER_FMT)
PACKET_HEADER_SIZE: Final[int] = struct.calcsize(_PACKET_HEADER_FMT)

# The C++ code depends on these exact sizes via sizeof().
assert GLOBAL_HEADER_SIZE == 24, f"global header must be 24 bytes, got {GLOBAL_HEADER_SIZE}"
assert PACKET_HEADER_SIZE == 16, f"packet header must be 16 bytes, got {PACKET_HEADER_SIZE}"

_UINT16_MASK: Final[int] = 0xFFFF
_UINT32_MASK: Final[int] = 0xFFFFFFFF

#: Upper bound on a single captured packet, from the C++ sanity check.
_MAX_PACKET_LENGTH: Final[int] = 65535


# ============================================================================
# PCAP Global Header (24 bytes) — at the very beginning of every .pcap file
# ============================================================================
@dataclass(slots=True)
class PcapGlobalHeader:
    """The 24-byte file header.

    Mirrors ``struct PcapGlobalHeader``.  ``thiszone`` is signed (``int32_t``);
    every other field is unsigned.
    """

    magic_number: int = 0  # 0xa1b2c3d4 (or swapped for big-endian)
    version_major: int = 0  # Usually 2
    version_minor: int = 0  # Usually 4
    thiszone: int = 0  # GMT offset (usually 0)
    sigfigs: int = 0  # Accuracy of timestamps (usually 0)
    snaplen: int = 0  # Max length of captured packets
    network: int = 0  # Data link type (1 = Ethernet)

    @classmethod
    def unpack(cls, raw: bytes) -> "PcapGlobalHeader":
        """Decode 24 bytes in native order, as the C++ raw struct read did."""
        return cls(*struct.unpack(_GLOBAL_HEADER_FMT, raw))


# ============================================================================
# PCAP Packet Header (16 bytes) — precedes each packet's bytes
# ============================================================================
@dataclass(slots=True)
class PcapPacketHeader:
    """The 16-byte per-packet record header.

    Mirrors ``struct PcapPacketHeader``.
    """

    ts_sec: int = 0  # Timestamp seconds
    ts_usec: int = 0  # Timestamp microseconds
    incl_len: int = 0  # Number of bytes saved in file
    orig_len: int = 0  # Actual length of packet

    @classmethod
    def unpack(cls, raw: bytes) -> "PcapPacketHeader":
        """Decode 16 bytes in native order, as the C++ raw struct read did."""
        return cls(*struct.unpack(_PACKET_HEADER_FMT, raw))


# ============================================================================
# Represents a single captured packet
# ============================================================================
@dataclass(slots=True)
class RawPacket:
    """A captured packet: its record header plus the raw bytes.

    Mirrors ``struct RawPacket``.  ``data`` replaces
    ``std::vector<uint8_t>``; ``len(data) == header.incl_len`` always holds
    for a packet returned by :meth:`PcapReader.read_next_packet`.
    """

    header: PcapPacketHeader = field(default_factory=PcapPacketHeader)
    data: bytes = b""  # The actual packet bytes


# ============================================================================
# Class to read PCAP files
# ============================================================================
class PcapReader:
    """Sequential reader for classic libpcap capture files.

    Mirrors ``class PcapReader``.  Usage as a context manager stands in for the
    C++ destructor::

        with PcapReader() as reader:
            if not reader.open("capture.pcap"):
                return 1
            for raw in reader:
                ...
    """

    __slots__ = ("_file", "_global_header", "_needs_byte_swap", "_filename", "_nanosecond")

    def __init__(self) -> None:
        self._file: BinaryIO | None = None
        self._global_header = PcapGlobalHeader()
        self._needs_byte_swap: bool = False
        self._filename: str = ""
        self._nanosecond: bool = False

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------
    def open(self, filename: str | Path) -> bool:
        """Open a pcap file for reading.

        Returns ``True`` on success.  Diagnostics and the informational banner
        are written to stderr/stdout with the same wording as the C++ original,
        because downstream tools compare console output.
        """
        # Close any previously opened file
        self.close()

        self._filename = str(filename)
        self._nanosecond = False
        path = Path(filename)

        # Open in binary mode - this is crucial for reading raw bytes
        try:
            self._file = path.open("rb")
        except OSError:
            # C++: file_.is_open() == false
            print(f"Error: Could not open file: {self._filename}", file=sys.stderr)
            return False

        # Read the global header (first 24 bytes of the file)
        raw = self._file.read(GLOBAL_HEADER_SIZE)
        if len(raw) != GLOBAL_HEADER_SIZE:
            # C++: !file_.good() after a short read
            print("Error: Could not read PCAP global header", file=sys.stderr)
            self.close()
            return False

        self._global_header = PcapGlobalHeader.unpack(raw)

        # Check the magic number to determine byte order
        if self._global_header.magic_number in (PCAP_MAGIC_NATIVE, PCAP_MAGIC_NATIVE_NS):
            self._needs_byte_swap = False
            self._nanosecond = self._global_header.magic_number == PCAP_MAGIC_NATIVE_NS
        elif self._global_header.magic_number in (PCAP_MAGIC_SWAPPED, PCAP_MAGIC_SWAPPED_NS):
            self._needs_byte_swap = True
            self._nanosecond = self._global_header.magic_number == PCAP_MAGIC_SWAPPED_NS
            # Swap the header fields we've already read.
            # NOTE: matching the original, magic_number, thiszone and sigfigs
            # are deliberately left unswapped.
            self._global_header.version_major = self._maybe_swap16(
                self._global_header.version_major
            )
            self._global_header.version_minor = self._maybe_swap16(
                self._global_header.version_minor
            )
            self._global_header.snaplen = self._maybe_swap32(self._global_header.snaplen)
            self._global_header.network = self._maybe_swap32(self._global_header.network)
        else:
            # std::hex prints lowercase with no leading zeros.
            print(
                f"Error: Invalid PCAP magic number: 0x{self._global_header.magic_number:x}",
                file=sys.stderr,
            )
            self.close()
            return False

        print(f"Opened PCAP file: {self._filename}")
        print(f"  Version: {self._global_header.version_major}.{self._global_header.version_minor}")
        print(f"  Snaplen: {self._global_header.snaplen} bytes")
        print(
            f"  Link type: {self._global_header.network}"
            f"{' (Ethernet)' if self._global_header.network == 1 else ''}"
        )

        return True

    def close(self) -> None:
        """Close the file and reset the byte-swap flag.

        Mirrors ``PcapReader::close()``, including resetting
        ``needs_byte_swap_`` to ``false``.
        """
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
        self._needs_byte_swap = False

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def read_next_packet(self) -> RawPacket | None:
        """Read the next packet, or ``None`` when there are no more.

        Mirrors ``bool readNextPacket(RawPacket&)``: ``None`` corresponds to the
        ``false`` return, which the C++ callers use as the loop-termination
        condition.  Note that ``false`` there covers both clean end-of-file and
        a hard error (bad length, truncated data), and both stop the read loop —
        preserved here.
        """
        if self._file is None:
            # C++: `if (!file_.is_open()) return false;`
            return None

        # Read the packet header (16 bytes)
        raw_header = self._file.read(PACKET_HEADER_SIZE)
        if len(raw_header) != PACKET_HEADER_SIZE:
            # End of file or error (C++: !file_.good(), reported silently)
            return None

        header = PcapPacketHeader.unpack(raw_header)

        # Swap bytes if needed
        if self._needs_byte_swap:
            header.ts_sec = self._maybe_swap32(header.ts_sec)
            header.ts_usec = self._maybe_swap32(header.ts_usec)
            header.incl_len = self._maybe_swap32(header.incl_len)
            header.orig_len = self._maybe_swap32(header.orig_len)

        # Sanity check on packet length
        if header.incl_len > self._global_header.snaplen or header.incl_len > _MAX_PACKET_LENGTH:
            print(f"Error: Invalid packet length: {header.incl_len}", file=sys.stderr)
            return None

        # Read the packet data
        data = self._file.read(header.incl_len)
        if len(data) != header.incl_len:
            print("Error: Could not read packet data", file=sys.stderr)
            return None

        return RawPacket(header=header, data=data)

    def __iter__(self) -> Iterator[RawPacket]:
        """Iterate packets until :meth:`read_next_packet` reports exhaustion."""
        while True:
            packet = self.read_next_packet()
            if packet is None:
                return
            yield packet

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_global_header(self) -> PcapGlobalHeader:
        """Return the global header (``getGlobalHeader()``)."""
        return self._global_header

    def is_open(self) -> bool:
        """Return whether the file is currently open (``isOpen()``)."""
        return self._file is not None and not self._file.closed

    def needs_byte_swap(self) -> bool:
        """Return whether file fields require byte swapping (``needsByteSwap()``)."""
        return self._needs_byte_swap

    def is_nanosecond(self) -> bool:
        """Return whether timestamps are nanoseconds rather than microseconds.

        No C++ counterpart — the original rejected nanosecond captures.  The
        sub-second field in ``PcapPacketHeader.ts_usec`` holds nanoseconds when
        this is ``True``; the reader does not rescale it, so a consumer that
        renders fractional seconds should divide by 1000 accordingly.
        """
        return self._nanosecond

    # ------------------------------------------------------------------
    # Byte-swap helpers
    # ------------------------------------------------------------------
    def _maybe_swap16(self, value: int) -> int:
        """Mirror ``PcapReader::maybeSwap16`` — a no-op unless swapping is on."""
        if not self._needs_byte_swap:
            return value & _UINT16_MASK
        value &= _UINT16_MASK
        return ((value & 0xFF00) >> 8) | ((value & 0x00FF) << 8)

    def _maybe_swap32(self, value: int) -> int:
        """Mirror ``PcapReader::maybeSwap32`` — a no-op unless swapping is on."""
        if not self._needs_byte_swap:
            return value & _UINT32_MASK
        value &= _UINT32_MASK
        return (
            ((value & 0xFF000000) >> 24)
            | ((value & 0x00FF0000) >> 8)
            | ((value & 0x0000FF00) << 8)
            | ((value & 0x000000FF) << 24)
        ) & _UINT32_MASK

    # ------------------------------------------------------------------
    # Context manager / cleanup (stands in for the C++ destructor)
    # ------------------------------------------------------------------
    def __enter__(self) -> "PcapReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        # ``~PcapReader() { close(); }``.  CPython refcounting makes this fire
        # promptly in practice, but it is a backstop: prefer the context
        # manager or an explicit close().
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "open" if self.is_open() else "closed"
        return f"PcapReader({self._filename!r}, {state}, swap={self._needs_byte_swap})"


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    if len(sys.argv) < 2:
        print(f"Usage: python -m dpi.pcap_reader <pcap_file>", file=sys.stderr)
        raise SystemExit(1)

    with PcapReader() as reader:
        if not reader.open(sys.argv[1]):
            raise SystemExit(1)

        total_packets = 0
        total_bytes = 0
        for raw in reader:
            total_packets += 1
            total_bytes += len(raw.data)
            assert len(raw.data) == raw.header.incl_len

        print(f"\nTotal packets read:  {total_packets}")
        print(f"Total bytes read:    {total_bytes}")
        print(f"Byte swap required:  {reader.needs_byte_swap()}")
        print(f"Nanosecond stamps:   {reader.is_nanosecond()}")
