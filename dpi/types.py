"""Core DPI data types.

Python port of ``include/types.h`` + ``src/types.cpp`` (C++ ``namespace DPI``).

This module is the foundation every other DPI component imports.  It defines
the flow identity (:class:`FiveTuple` and its load-balancing hash), the
application taxonomy (:class:`AppType` and the SNI classifier), connection and
packet records, and the global statistics block.

C++ concepts replaced
---------------------
``struct FiveTuple`` with ``operator==`` and ``FiveTupleHash``
    Becomes a frozen, slotted dataclass.  ``__eq__``/``__hash__`` are generated
    for use as a ``dict`` key (the Python analogue of
    ``unordered_map<FiveTuple, Connection, FiveTupleHash>``).  The *exact*
    64-bit boost-style hash is kept separately as :func:`five_tuple_hash`,
    because the load balancer computes ``hash % num_fps`` and any change to the
    hash would change which FP thread a flow is pinned to.

``enum class``
    Becomes :class:`enum.IntEnum`, which keeps the implicit integer values
    (``AppType::UNKNOWN == 0``, ``APP_COUNT`` last) while staying hashable for
    use in sets, as ``blocked_apps_`` requires.

``std::chrono::steady_clock::time_point``
    Becomes a ``float`` from :func:`time.monotonic`.  A default-constructed
    C++ ``time_point`` sits at the clock epoch, so the default here is ``0.0``.

``std::atomic<uint64_t>``
    Becomes :class:`AtomicCounter`, an explicitly mutex-guarded counter.
    ``int += 1`` is *not* atomic in CPython once the GIL is removed
    (PEP 703 free-threaded builds), so relying on it would be a real race on
    Python 3.13+ ``--disable-gil`` and is wrong in principle even under the GIL.

``const uint8_t* payload_data`` (borrowed pointer into the packet buffer)
    Becomes a :class:`memoryview` slice, which is likewise a zero-copy view
    over the owning ``bytes`` buffer rather than a duplicate of the payload.

Fixed upstream bugs
-------------------
``sni_to_app_type`` no longer reproduces the three misclassifications present
in the C++ original — ``www.netflix.com`` and ``raw.githubusercontent.com``
classifying as Twitter/X, and ``yt3.ggpht.com`` as Google.  See the comment
above :data:`_SNI_PATTERNS` for what changed and why.  Classification counts
therefore differ from the C++ build **by design**.

Note
----
This module keeps the original file name ``types.py``, which collides with the
standard library :mod:`types`.  Absolute imports keep that harmless for normal
use, but run the self-test as ``python -m dpi.types`` — running
``python dpi/types.py`` puts ``dpi/`` on ``sys.path[0]`` and shadows the stdlib
module that :mod:`enum` and :mod:`functools` depend on.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final

__all__ = [
    "UINT64_MASK",
    "GOLDEN_RATIO_CONSTANT",
    "FiveTuple",
    "five_tuple_hash",
    "flow_hash",
    "FiveTupleHash",
    "AppType",
    "app_type_to_string",
    "sni_to_app_type",
    "MatchMode",
    "ConnectionState",
    "PacketAction",
    "Connection",
    "PacketJob",
    "AtomicCounter",
    "DPIStats",
    # C++-style aliases
    "appTypeToString",
    "sniToAppType",
]

#: Mask emulating ``size_t`` truncation on a 64-bit platform.  The C++ hash
#: relies on unsigned wrap-around; Python ints are arbitrary precision.
UINT64_MASK: Final[int] = 0xFFFFFFFFFFFFFFFF

#: 2^32 / golden ratio — the boost ``hash_combine`` magic constant, verbatim
#: from the C++ ``FiveTupleHash``.
GOLDEN_RATIO_CONSTANT: Final[int] = 0x9E3779B9


# ============================================================================
# Five-Tuple: Uniquely identifies a connection/flow
# ============================================================================
@dataclass(frozen=True, slots=True)
class FiveTuple:
    """Uniquely identifies a connection/flow.

    Field widths mirror the C++ struct: ``src_ip``/``dst_ip`` are ``uint32_t``,
    ``src_port``/``dst_port`` are ``uint16_t``, ``protocol`` is ``uint8_t``
    (TCP=6, UDP=17).

    IP addresses are stored **exactly as they appear on the wire**, i.e. the
    raw 4 bytes read into a native-order integer.  On a little-endian host that
    means the first octet occupies the *low* byte — which is why
    :meth:`to_string` shifts by 0/8/16/24 in that order, matching
    ``FiveTuple::toString()`` in ``types.cpp``.
    """

    src_ip: int
    dst_ip: int
    src_port: int
    dst_port: int
    protocol: int

    def reverse(self) -> "FiveTuple":
        """Create the reverse tuple, for matching bidirectional flows.

        Mirrors ``FiveTuple::reverse()``::

            return {dst_ip, src_ip, dst_port, src_port, protocol};
        """
        return FiveTuple(
            src_ip=self.dst_ip,
            dst_ip=self.src_ip,
            src_port=self.dst_port,
            dst_port=self.src_port,
            protocol=self.protocol,
        )

    @staticmethod
    def format_ip(ip: int) -> str:
        """Format a wire-order IPv4 word as dotted quad.

        Mirrors the ``formatIP`` lambda inside ``FiveTuple::toString()``, which
        emits ``(ip >> 0) & 0xFF`` first.  Deliberately *not*
        :func:`ipaddress.IPv4Address`, whose big-endian interpretation would
        print the octets reversed for these values.
        """
        return (
            f"{(ip >> 0) & 0xFF}."
            f"{(ip >> 8) & 0xFF}."
            f"{(ip >> 16) & 0xFF}."
            f"{(ip >> 24) & 0xFF}"
        )

    def to_string(self) -> str:
        """Human-readable flow description.

        Mirrors ``FiveTuple::toString()`` byte for byte, e.g.::

            192.168.1.10:54321 -> 142.250.183.14:443 (TCP)
        """
        if self.protocol == 6:
            proto = "TCP"
        elif self.protocol == 17:
            proto = "UDP"
        else:
            proto = "?"

        return (
            f"{self.format_ip(self.src_ip)}:{self.src_port}"
            f" -> "
            f"{self.format_ip(self.dst_ip)}:{self.dst_port}"
            f" ({proto})"
        )

    def canonical(self) -> "FiveTuple":
        """Return a direction-independent form of this tuple.

        The two endpoints are ordered, so a flow and its reverse both map to
        the same value.  No C++ counterpart — it is what makes
        :func:`flow_hash` symmetric, which in turn is what lets both directions
        of a conversation reach the same FP thread and share one
        :class:`Connection`.
        """
        if (self.src_ip, self.src_port) <= (self.dst_ip, self.dst_port):
            return self
        return self.reverse()

    def hash_value(self) -> int:
        """Exact 64-bit ``FiveTupleHash`` value (see :func:`five_tuple_hash`)."""
        return five_tuple_hash(self)

    def __str__(self) -> str:
        return self.to_string()


def five_tuple_hash(tuple_: FiveTuple) -> int:
    """Reproduce ``DPI::FiveTupleHash::operator()`` exactly.

    The C++ original is a boost-style ``hash_combine`` chain::

        size_t h = 0;
        h ^= std::hash<uint32_t>{}(tuple.src_ip) + 0x9e3779b9 + (h << 6) + (h >> 2);
        ... (dst_ip, src_port, dst_port, protocol)
        return h;

    Two details that must be preserved bit-for-bit, because this value decides
    which FP thread owns a flow:

    * ``std::hash`` of an integral type no wider than ``size_t`` is the
      *identity* function in libstdc++ and MSVC, so each term contributes its
      raw field value.
    * ``h`` on the right-hand side is the value from *before* the compound
      assignment, and every term wraps modulo 2^64.  ``>>`` is a logical shift
      because ``size_t`` is unsigned; Python's ``>>`` on a non-negative int
      matches that.
    """
    h = 0
    for value in (
        tuple_.src_ip,
        tuple_.dst_ip,
        tuple_.src_port,
        tuple_.dst_port,
        tuple_.protocol,
    ):
        term = (value + GOLDEN_RATIO_CONSTANT + (h << 6) + (h >> 2)) & UINT64_MASK
        h = (h ^ term) & UINT64_MASK
    return h


def flow_hash(tuple_: FiveTuple) -> int:
    """Direction-independent flow hash: ``five_tuple_hash(tuple.canonical())``.

    Used for load balancing so that a conversation's two directions are pinned
    to the same LB and the same FP.  The C++ hashed the raw tuple, which sent
    each direction to a different FP — see
    :meth:`~dpi.connection_tracker.ConnectionTracker.get_or_create_connection`.
    """
    return five_tuple_hash(tuple_.canonical())


class FiveTupleHash:
    """Callable hash functor, mirroring the C++ ``struct FiveTupleHash``.

    Kept as a class so ported call sites can read like the original
    (``FiveTupleHash()(tuple)``); :func:`five_tuple_hash` is the direct form.
    """

    __slots__ = ()

    def __call__(self, tuple_: FiveTuple) -> int:
        return five_tuple_hash(tuple_)


# ============================================================================
# Application Classification
# ============================================================================
class AppType(IntEnum):
    """Application classification.

    Values mirror ``enum class AppType`` in ``types.h``, including the trailing
    ``APP_COUNT`` sentinel used for sizing arrays in the C++ code.
    """

    UNKNOWN = 0
    HTTP = 1
    HTTPS = 2
    DNS = 3
    TLS = 4
    QUIC = 5
    # Specific applications (detected via SNI)
    GOOGLE = 6
    FACEBOOK = 7
    YOUTUBE = 8
    TWITTER = 9
    INSTAGRAM = 10
    NETFLIX = 11
    AMAZON = 12
    MICROSOFT = 13
    APPLE = 14
    WHATSAPP = 15
    TELEGRAM = 16
    TIKTOK = 17
    SPOTIFY = 18
    ZOOM = 19
    DISCORD = 20
    GITHUB = 21
    CLOUDFLARE = 22
    APP_COUNT = 23  # Keep this last for counting


#: Display names, mirroring the ``switch`` in ``appTypeToString()``.
_APP_TYPE_NAMES: Final[dict[AppType, str]] = {
    AppType.UNKNOWN: "Unknown",
    AppType.HTTP: "HTTP",
    AppType.HTTPS: "HTTPS",
    AppType.DNS: "DNS",
    AppType.TLS: "TLS",
    AppType.QUIC: "QUIC",
    AppType.GOOGLE: "Google",
    AppType.FACEBOOK: "Facebook",
    AppType.YOUTUBE: "YouTube",
    AppType.TWITTER: "Twitter/X",
    AppType.INSTAGRAM: "Instagram",
    AppType.NETFLIX: "Netflix",
    AppType.AMAZON: "Amazon",
    AppType.MICROSOFT: "Microsoft",
    AppType.APPLE: "Apple",
    AppType.WHATSAPP: "WhatsApp",
    AppType.TELEGRAM: "Telegram",
    AppType.TIKTOK: "TikTok",
    AppType.SPOTIFY: "Spotify",
    AppType.ZOOM: "Zoom",
    AppType.DISCORD: "Discord",
    AppType.GITHUB: "GitHub",
    AppType.CLOUDFLARE: "Cloudflare",
}


def app_type_to_string(type_: AppType) -> str:
    """Return the display name for an :class:`AppType`.

    Mirrors ``appTypeToString()``, including its ``default: return "Unknown"``
    fall-through (which is what ``APP_COUNT`` and any out-of-range value hit).
    """
    return _APP_TYPE_NAMES.get(type_, "Unknown")


# ---------------------------------------------------------------------------
# SNI -> AppType classification table
# ---------------------------------------------------------------------------
# FIXED (was UPSTREAM BUG): the C++ used plain substring tests for every
# pattern and ordered general groups before specific ones, which produced three
# reproducible misclassifications:
#
#     www.netflix.com           -> TWITTER  (Twitter's "x.com" hit "netfli|x.com|")
#     raw.githubusercontent.com -> TWITTER  (Twitter's "t.co" hit "...conten|t.co|m")
#     yt3.ggpht.com             -> GOOGLE   (Google tested before YouTube)
#
# Two changes fix all three without narrowing legitimate coverage:
#
#   1. Each pattern now carries a MATCH MODE, so short domain-shaped patterns
#      ("x.com", "t.co") only match at a label boundary, and ambiguous bare
#      words ("bing", "aws", "apple") only match a whole label.  Long
#      distinctive tokens ("googleapis", "githubusercontent") keep substring
#      matching, which is what gives broad CDN coverage.
#   2. Specific brands are ordered before the umbrella brands that would
#      otherwise swallow them (YouTube before Google, Netflix/GitHub before
#      Twitter).
#
# Evaluation still stops at the first group that matches, so order remains
# meaningful -- it is now correct rather than accidental.


class MatchMode(IntEnum):
    """How a classification pattern is compared against a hostname."""

    SUB = 0  # substring anywhere -- for long, unambiguous tokens
    DOM = 1  # domain suffix -- host == p, or host ends with "." + p
    LBL = 2  # one whole DNS label equals p
    PFX = 3  # some DNS label starts with p


_SUB: Final[MatchMode] = MatchMode.SUB
_DOM: Final[MatchMode] = MatchMode.DOM
_LBL: Final[MatchMode] = MatchMode.LBL
_PFX: Final[MatchMode] = MatchMode.PFX


def _pattern_matches(host: str, pattern: str, mode: MatchMode) -> bool:
    """Test one lowercased hostname against one lowercased pattern."""
    if mode == MatchMode.SUB:
        return pattern in host
    if mode == MatchMode.DOM:
        return host == pattern or host.endswith("." + pattern)
    labels = host.split(".")
    if mode == MatchMode.LBL:
        return pattern in labels
    return any(label.startswith(pattern) for label in labels)


_SNI_PATTERNS: Final[tuple[tuple[AppType, tuple[tuple[str, MatchMode], ...]], ...]] = (
    # --- specific brands first -------------------------------------------
    # YouTube before Google: Google's "ggpht" would otherwise claim yt3.ggpht.
    (
        AppType.YOUTUBE,
        (("youtube", _SUB), ("ytimg", _SUB), ("youtu.be", _DOM), ("yt3.ggpht", _SUB)),
    ),
    (AppType.INSTAGRAM, (("instagram", _SUB), ("cdninstagram", _SUB))),
    (AppType.WHATSAPP, (("whatsapp", _SUB), ("wa.me", _DOM))),
    (
        AppType.FACEBOOK,
        (("facebook", _SUB), ("fbcdn", _SUB), ("fb.com", _DOM),
         ("fbsbx", _SUB), ("meta.com", _DOM)),
    ),
    # Netflix and GitHub before Twitter, whose short patterns used to shadow them.
    (AppType.NETFLIX, (("netflix", _SUB), ("nflxvideo", _SUB), ("nflximg", _SUB))),
    (AppType.GITHUB, (("github", _SUB), ("githubusercontent", _SUB))),
    (AppType.TIKTOK, (("tiktok", _SUB), ("tiktokcdn", _SUB),
                      ("musical.ly", _DOM), ("bytedance", _SUB))),
    (AppType.TELEGRAM, (("telegram", _SUB), ("t.me", _DOM))),
    (AppType.SPOTIFY, (("spotify", _SUB), ("scdn.co", _DOM))),
    (AppType.DISCORD, (("discord", _SUB), ("discordapp", _SUB))),
    (AppType.ZOOM, (("zoom", _LBL),)),
    # "x.com"/"t.co" are domain-anchored, so they can no longer match mid-string.
    (AppType.TWITTER, (("twitter", _SUB), ("twimg", _SUB),
                       ("x.com", _DOM), ("t.co", _DOM))),
    # --- umbrella brands last --------------------------------------------
    (
        AppType.GOOGLE,
        (("google", _SUB), ("gstatic", _SUB), ("googleapis", _SUB),
         ("ggpht", _SUB), ("gvt1", _SUB)),
    ),
    (
        AppType.AMAZON,
        (("amazon", _SUB), ("amazonaws", _SUB), ("cloudfront", _SUB), ("aws", _LBL)),
    ),
    (
        AppType.MICROSOFT,
        (("microsoft", _SUB), ("msn.com", _DOM), ("office", _PFX), ("azure", _SUB),
         ("live.com", _DOM), ("outlook", _SUB), ("bing", _LBL)),
    ),
    (
        AppType.APPLE,
        (("apple", _LBL), ("icloud", _SUB), ("mzstatic", _SUB), ("itunes", _SUB)),
    ),
    (AppType.CLOUDFLARE, (("cloudflare", _SUB), ("cf-", _PFX))),
)


def sni_to_app_type(sni: str) -> AppType:
    """Map an SNI / domain name to an :class:`AppType`.

    Mirrors ``sniToAppType()``: lowercase the input, then walk the pattern
    groups in declaration order and return on the first substring hit.

    An empty name yields ``UNKNOWN``; a non-empty but unrecognised name yields
    ``HTTPS``, matching the C++ trailing comment "If SNI is present but not
    recognized, still mark as TLS/HTTPS".
    """
    if not sni:
        return AppType.UNKNOWN

    # std::transform(..., ::tolower) over the whole string.
    lower_sni = sni.lower()

    for app_type, patterns in _SNI_PATTERNS:
        for pattern, mode in patterns:
            if _pattern_matches(lower_sni, pattern, mode):
                return app_type

    return AppType.HTTPS


# ============================================================================
# Connection State
# ============================================================================
class ConnectionState(IntEnum):
    """Lifecycle state of a tracked connection."""

    NEW = 0
    ESTABLISHED = 1
    CLASSIFIED = 2
    BLOCKED = 3
    CLOSED = 4


# ============================================================================
# Packet Action (what to do with the packet)
# ============================================================================
class PacketAction(IntEnum):
    """Verdict for a packet."""

    FORWARD = 0  # Send to internet
    DROP = 1  # Block/drop the packet
    INSPECT = 2  # Needs further inspection
    LOG_ONLY = 3  # Forward but log


# ============================================================================
# Connection Entry (tracked per flow)
# ============================================================================
@dataclass(slots=True)
class Connection:
    """Per-flow tracking record.

    Mutable by design — the FP thread updates counters and state in place, the
    same way the C++ code mutates the ``Connection&`` it holds a pointer to.

    ``first_seen`` / ``last_seen`` are monotonic-clock seconds
    (:func:`time.monotonic`), standing in for
    ``std::chrono::steady_clock::time_point``.  They default to ``0.0``
    because a default-constructed C++ ``time_point`` sits at the clock epoch;
    the tracker stamps them on insert.
    """

    tuple: FiveTuple
    state: ConnectionState = ConnectionState.NEW
    app_type: AppType = AppType.UNKNOWN
    sni: str = ""  # Server Name Indication (if detected)

    packets_in: int = 0
    packets_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0

    first_seen: float = 0.0
    last_seen: float = 0.0

    action: PacketAction = PacketAction.FORWARD

    # For TCP state tracking
    syn_seen: bool = False
    syn_ack_seen: bool = False
    fin_seen: bool = False


# ============================================================================
# Packet wrapper for queue passing
# ============================================================================
@dataclass(slots=True)
class PacketJob:
    """A packet in flight between Reader, LB and FP threads.

    ``data`` owns the packet bytes (the C++ ``std::vector<uint8_t>``), and the
    offsets index into it.  ``payload_data`` is a borrowed zero-copy
    :class:`memoryview`, standing in for the C++ ``const uint8_t*`` that
    pointed into that same buffer — assigning it does not copy the payload.
    """

    packet_id: int = 0
    tuple: FiveTuple | None = None
    data: bytes = b""
    eth_offset: int = 0
    ip_offset: int = 0
    transport_offset: int = 0
    payload_offset: int = 0
    payload_length: int = 0
    tcp_flags: int = 0
    payload_data: memoryview | None = None

    # Timestamps
    ts_sec: int = 0
    ts_usec: int = 0

    def get_payload(self) -> memoryview:
        """Return a zero-copy view of the payload bytes.

        Uses ``payload_data`` when the parser already set it, otherwise slices
        ``data`` at the recorded offset — the two are equivalent, exactly as
        the C++ pointer and ``data.data() + payload_offset`` were.
        """
        if self.payload_data is not None:
            return self.payload_data
        return memoryview(self.data)[
            self.payload_offset : self.payload_offset + self.payload_length
        ]


# ============================================================================
# Statistics
# ============================================================================
class AtomicCounter:
    """A mutex-guarded 64-bit counter, standing in for ``std::atomic<uint64_t>``.

    CPython's ``+=`` on an ``int`` attribute is a load-add-store that the
    interpreter may interrupt, and on free-threaded builds (PEP 703) there is
    no GIL serialising it at all.  An explicit lock keeps increments correct on
    every build, which is what the C++ ``std::atomic`` guaranteed.
    """

    __slots__ = ("_value", "_lock")

    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._lock = threading.Lock()

    def add(self, delta: int = 1) -> int:
        """Atomically add ``delta`` and return the new value (``fetch_add`` + 1)."""
        with self._lock:
            self._value = (self._value + delta) & UINT64_MASK
            return self._value

    def increment(self) -> int:
        """Atomically add one and return the new value (``operator++``)."""
        return self.add(1)

    def get(self) -> int:
        """Atomically read the current value (``load()``)."""
        with self._lock:
            return self._value

    def set(self, value: int) -> None:
        """Atomically overwrite the value (``store()``)."""
        with self._lock:
            self._value = value & UINT64_MASK

    def reset(self) -> None:
        """Set the counter back to zero."""
        self.set(0)

    def __int__(self) -> int:
        return self.get()

    def __index__(self) -> int:
        return self.get()

    def __repr__(self) -> str:
        return f"AtomicCounter({self.get()})"


@dataclass(slots=True)
class DPIStats:
    """Engine-wide counters.

    The C++ struct deleted its copy constructor and assignment operator because
    ``std::atomic`` members are non-copyable.  The Python analogue is that
    :class:`AtomicCounter` holds a :class:`threading.Lock`, which likewise
    cannot be meaningfully copied — use :meth:`snapshot` to read a consistent
    set of plain ints instead of copying the object.
    """

    total_packets: AtomicCounter = field(default_factory=AtomicCounter)
    total_bytes: AtomicCounter = field(default_factory=AtomicCounter)
    forwarded_packets: AtomicCounter = field(default_factory=AtomicCounter)
    dropped_packets: AtomicCounter = field(default_factory=AtomicCounter)
    tcp_packets: AtomicCounter = field(default_factory=AtomicCounter)
    udp_packets: AtomicCounter = field(default_factory=AtomicCounter)
    other_packets: AtomicCounter = field(default_factory=AtomicCounter)
    active_connections: AtomicCounter = field(default_factory=AtomicCounter)

    def snapshot(self) -> dict[str, int]:
        """Read every counter into a plain ``dict`` of ints, for reporting."""
        return {
            "total_packets": self.total_packets.get(),
            "total_bytes": self.total_bytes.get(),
            "forwarded_packets": self.forwarded_packets.get(),
            "dropped_packets": self.dropped_packets.get(),
            "tcp_packets": self.tcp_packets.get(),
            "udp_packets": self.udp_packets.get(),
            "other_packets": self.other_packets.get(),
            "active_connections": self.active_connections.get(),
        }

    def reset(self) -> None:
        """Zero every counter."""
        for counter in (
            self.total_packets,
            self.total_bytes,
            self.forwarded_packets,
            self.dropped_packets,
            self.tcp_packets,
            self.udp_packets,
            self.other_packets,
            self.active_connections,
        ):
            counter.reset()


# ---------------------------------------------------------------------------
# camelCase aliases matching the original C++ symbol names.
# ---------------------------------------------------------------------------
appTypeToString = app_type_to_string
sniToAppType = sni_to_app_type


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    # FiveTuple: wire-order IP formatting and reversal
    # 192.168.1.10 on the wire -> bytes C0 A8 01 0A -> 0x0A01A8C0 little-endian
    t = FiveTuple(
        src_ip=0x0A01A8C0,
        dst_ip=0x0EB7FA8E,
        src_port=54321,
        dst_port=443,
        protocol=6,
    )
    print(t.to_string())
    assert t.to_string() == "192.168.1.10:54321 -> 142.250.183.14:443 (TCP)"

    r = t.reverse()
    assert r.src_ip == t.dst_ip and r.dst_port == t.src_port
    assert r.reverse() == t
    print(f"reverse(): {r.to_string()}")

    # Canonical form + flow hash are direction-independent
    assert t.canonical() == r.canonical()
    assert flow_hash(t) == flow_hash(r), "both directions must share a flow hash"

    # Hash must be stable, non-zero, and order-sensitive
    h = five_tuple_hash(t)
    assert h == FiveTupleHash()(t) == t.hash_value()
    assert 0 <= h <= UINT64_MASK
    assert five_tuple_hash(t) != five_tuple_hash(r)
    print(f"five_tuple_hash = 0x{h:016X}  -> FP {h % 4} of 4")

    # Usable as a dict key (unordered_map<FiveTuple, Connection, FiveTupleHash>)
    table: dict[FiveTuple, Connection] = {t: Connection(tuple=t)}
    assert table[FiveTuple(0x0A01A8C0, 0x0EB7FA8E, 54321, 443, 6)].tuple == t

    # SNI classification, including the two preserved quirks
    assert sni_to_app_type("") is AppType.UNKNOWN
    assert sni_to_app_type("www.youtube.com") is AppType.YOUTUBE
    assert sni_to_app_type("WWW.GOOGLE.COM") is AppType.GOOGLE
    assert sni_to_app_type("scontent.fbcdn.net") is AppType.FACEBOOK
    assert sni_to_app_type("example.invalid") is AppType.HTTPS
    # Regression guards for the three fixed misclassifications.
    assert sni_to_app_type("www.netflix.com") is AppType.NETFLIX
    assert sni_to_app_type("raw.githubusercontent.com") is AppType.GITHUB
    assert sni_to_app_type("yt3.ggpht.com") is AppType.YOUTUBE
    assert sni_to_app_type("abbot.com") is AppType.HTTPS       # was Twitter/X
    assert sni_to_app_type("tubing.net") is AppType.HTTPS      # "bing" no longer hits
    assert sni_to_app_type("pineapple.com") is AppType.HTTPS   # "apple" no longer hits
    # ...while real coverage is unchanged:
    assert sni_to_app_type("x.com") is AppType.TWITTER
    assert sni_to_app_type("t.co") is AppType.TWITTER
    assert sni_to_app_type("api.twitter.com") is AppType.TWITTER
    assert sni_to_app_type("s3.amazonaws.com") is AppType.AMAZON
    assert sni_to_app_type("aws.amazon.com") is AppType.AMAZON
    assert sni_to_app_type("outlook.office365.com") is AppType.MICROSOFT
    assert sni_to_app_type("www.bing.com") is AppType.MICROSOFT
    assert sni_to_app_type("www.apple.com") is AppType.APPLE
    assert sni_to_app_type("us02web.zoom.us") is AppType.ZOOM
    assert sni_to_app_type("i.scdn.co") is AppType.SPOTIFY
    assert sni_to_app_type("fonts.gstatic.com") is AppType.GOOGLE

    assert app_type_to_string(AppType.TWITTER) == "Twitter/X"
    assert app_type_to_string(AppType.APP_COUNT) == "Unknown"

    # Atomic counters
    stats = DPIStats()
    threads = [
        threading.Thread(target=lambda: [stats.total_packets.increment() for _ in range(1000)])
        for _ in range(8)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert stats.total_packets.get() == 8000, stats.total_packets.get()
    print("DPIStats after 8x1000 concurrent increments:", stats.snapshot()["total_packets"])

    print("types.py self-test OK")
