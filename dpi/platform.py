"""Portable byte-order conversion helpers.

Python port of ``include/platform.h``.

The original C++ header lived in namespace ``PortableNet`` and provided
byte-order conversion routines that deliberately avoided ``<arpa/inet.h>`` /
``<winsock2.h>`` so the project would build unchanged on Linux, macOS and
Windows.  This module reproduces those routines exactly, including their
unsigned 16/32-bit wrap-around semantics.

C++ concepts replaced
---------------------
``inline`` free functions in a namespace
    Become module-level functions.  ``from dpi import platform as PortableNet``
    reproduces the original call syntax (``PortableNet.net_to_host16(...)``).

``reinterpret_cast<uint8_t*>(&test)`` endianness probe
    Python has no addressable stack objects, so the equivalent probe packs a
    known value with :mod:`struct` in native order and inspects byte 0.  The
    result is identical to the C++ trick and is additionally cross-checked
    against :data:`sys.byteorder`.

``uint16_t`` / ``uint32_t`` fixed-width wrap-around
    Python ``int`` is arbitrary precision, so every operation is masked with
    ``0xFFFF`` / ``0xFFFFFFFF`` to keep the C++ truncation behaviour.

Note
----
The module is named ``platform`` for parity with the original file name.
Because Python 3 uses absolute imports, this does *not* shadow the standard
library :mod:`platform` module for any other code; inside this package it is
always reached as ``dpi.platform``.
"""

from __future__ import annotations

import struct
import sys
from typing import Final

__all__ = [
    "UINT16_MASK",
    "UINT32_MASK",
    "swap_bytes16",
    "swap_bytes32",
    "is_little_endian",
    "net_to_host16",
    "net_to_host32",
    "host_to_net16",
    "host_to_net32",
    # C++-style aliases, so ported call sites can stay verbatim.
    "swapBytes16",
    "swapBytes32",
    "isLittleEndian",
    "netToHost16",
    "netToHost32",
    "hostToNet16",
    "hostToNet32",
]

# ---------------------------------------------------------------------------
# Fixed-width masks (emulate uint16_t / uint32_t truncation)
# ---------------------------------------------------------------------------
UINT16_MASK: Final[int] = 0xFFFF
UINT32_MASK: Final[int] = 0xFFFFFFFF


def swap_bytes16(value: int) -> int:
    """Reverse the two bytes of a 16-bit value.

    Mirrors::

        inline uint16_t swapBytes16(uint16_t value) {
            return ((value & 0xFF00) >> 8) | ((value & 0x00FF) << 8);
        }
    """
    value &= UINT16_MASK
    return ((value & 0xFF00) >> 8) | ((value & 0x00FF) << 8)


def swap_bytes32(value: int) -> int:
    """Reverse the four bytes of a 32-bit value.

    Mirrors::

        inline uint32_t swapBytes32(uint32_t value) {
            return ((value & 0xFF000000) >> 24) |
                   ((value & 0x00FF0000) >> 8)  |
                   ((value & 0x0000FF00) << 8)  |
                   ((value & 0x000000FF) << 24);
        }
    """
    value &= UINT32_MASK
    return (
        ((value & 0xFF000000) >> 24)
        | ((value & 0x00FF0000) >> 8)
        | ((value & 0x0000FF00) << 8)
        | ((value & 0x000000FF) << 24)
    ) & UINT32_MASK


def _probe_little_endian() -> bool:
    """Runtime endianness probe, equivalent to the C++ ``reinterpret_cast`` trick.

    The C++ original writes ``uint16_t test = 0x0001`` and reads its first byte.
    Packing ``0x0001`` in native order and checking byte 0 is the same test.
    """
    return struct.pack("=H", 0x0001)[0] == 0x01


#: Cached result of the endianness probe.  The C++ version re-ran the probe on
#: every call; the outcome cannot change during a process lifetime, so caching
#: preserves behaviour while avoiding per-packet overhead on the hot path.
_IS_LITTLE_ENDIAN: Final[bool] = _probe_little_endian()

# Defensive cross-check: the struct probe and sys.byteorder must agree.
assert _IS_LITTLE_ENDIAN == (sys.byteorder == "little"), (
    "Endianness probe disagrees with sys.byteorder; "
    "refusing to continue with an inconsistent byte-order model."
)


def is_little_endian() -> bool:
    """Return ``True`` when the host CPU is little-endian.

    Mirrors ``PortableNet::isLittleEndian()``.
    """
    return _IS_LITTLE_ENDIAN


def net_to_host16(net_value: int) -> int:
    """Convert a 16-bit value from network (big-endian) to host byte order."""
    if _IS_LITTLE_ENDIAN:
        return swap_bytes16(net_value)
    return net_value & UINT16_MASK


def net_to_host32(net_value: int) -> int:
    """Convert a 32-bit value from network (big-endian) to host byte order."""
    if _IS_LITTLE_ENDIAN:
        return swap_bytes32(net_value)
    return net_value & UINT32_MASK


def host_to_net16(host_value: int) -> int:
    """Convert a 16-bit value from host to network byte order.

    Byte reversal is an involution, so this is the same operation as
    :func:`net_to_host16` — exactly as in the C++ header, where
    ``hostToNet16`` simply forwarded to ``netToHost16``.
    """
    return net_to_host16(host_value)


def host_to_net32(host_value: int) -> int:
    """Convert a 32-bit value from host to network byte order.

    Same involution argument as :func:`host_to_net16`.
    """
    return net_to_host32(host_value)


# ---------------------------------------------------------------------------
# camelCase aliases matching the original C++ symbol names.
# These let ported call sites read identically to the C++ source.
# ---------------------------------------------------------------------------
swapBytes16 = swap_bytes16
swapBytes32 = swap_bytes32
isLittleEndian = is_little_endian
netToHost16 = net_to_host16
netToHost32 = net_to_host32
hostToNet16 = host_to_net16
hostToNet32 = host_to_net32


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    print(f"Host is little-endian: {is_little_endian()}")
    print(f"swap_bytes16(0x1234)  = 0x{swap_bytes16(0x1234):04X}")
    print(f"swap_bytes32(0x12345678) = 0x{swap_bytes32(0x12345678):08X}")
    # A big-endian port field "00 50" read raw into a native uint16 on a
    # little-endian host looks like 0x5000; netToHost16 turns it back into 80.
    print(f"net_to_host16(0x5000) = {net_to_host16(0x5000)}  (expect 80)")
    print(f"net_to_host16(0xBB01) = {net_to_host16(0xBB01)}  (expect 443)")
    assert swap_bytes16(swap_bytes16(0xABCD)) == 0xABCD
    assert swap_bytes32(swap_bytes32(0xDEADBEEF)) == 0xDEADBEEF
    assert host_to_net16(net_to_host16(0x1234)) == 0x1234
    print("platform.py self-test OK")
