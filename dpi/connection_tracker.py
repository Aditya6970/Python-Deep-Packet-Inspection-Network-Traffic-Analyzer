"""Per-flow connection tracking and cross-FP aggregation.

Python port of ``include/connection_tracker.h`` + ``src/connection_tracker.cpp``
(C++ ``namespace DPI``).

Each FP thread owns one :class:`ConnectionTracker`.  No locking is needed
because consistent hashing pins a flow to a single FP, so only that thread ever
mutates its table.  :class:`GlobalConnectionTable` aggregates across trackers
for reporting.

Connection lifecycle::

    NEW -> ESTABLISHED -> CLASSIFIED -> CLOSED
                       \\-> BLOCKED

Fixed upstream bug
------------------
:meth:`ConnectionTracker.get_or_create_connection` now matches the reverse
five-tuple, so both directions of a conversation share one record.  In the C++
they did not, which inflated connection counts, made ``ESTABLISHED``
unreachable, and split each flow's classification in two.

C++ concepts replaced
---------------------
``Connection*`` returned from ``getOrCreateConnection``
    Becomes the :class:`~dpi.types.Connection` object itself.  Python objects
    are reference types, so the caller mutates the very object stored in the
    table — exactly what the raw pointer gave.  ``nullptr`` becomes ``None``.

``std::unordered_map<FiveTuple, Connection, FiveTupleHash>``
    Becomes a ``dict`` keyed on :class:`~dpi.types.FiveTuple`, which is a
    frozen dataclass and therefore hashable.

``std::chrono::steady_clock::now()``
    Becomes :func:`time.monotonic`.

``std::function<void(const Connection&)>`` callback
    Becomes any ``Callable[[Connection], None]``.

``std::shared_mutex`` in GlobalConnectionTable
    Becomes :class:`threading.Lock`; see the note in
    :mod:`dpi.rule_manager` on why the shared/exclusive distinction does not
    change semantics here.

Thread-safety note (a real difference, not a stylistic one)
-----------------------------------------------------------
``GlobalConnectionTable::getGlobalStats`` calls ``tracker->forEach()`` from the
reporting thread while the owning FP thread is still inserting into that same
``unordered_map``.  In C++ that is a **data race** — undefined behaviour that
usually appears to work and occasionally corrupts.  In Python the same pattern
raises ``RuntimeError: dictionary changed size during iteration``.  Since a
crash is not an acceptable translation of "usually works", :meth:`for_each` and
:meth:`get_all_connections` iterate over a snapshot taken with a single
``list()`` call, which is atomic in CPython.  The reader may see a slightly
stale view — which is all the C++ could honestly promise anyway.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Final

from .types import AppType, Connection, ConnectionState, FiveTuple, PacketAction, app_type_to_string

__all__ = [
    "DEFAULT_MAX_CONNECTIONS",
    "DEFAULT_STALE_TIMEOUT_SECONDS",
    "TrackerStats",
    "GlobalStats",
    "ConnectionTracker",
    "GlobalConnectionTable",
]

#: Default flow-table capacity, from the C++ default argument.
DEFAULT_MAX_CONNECTIONS: Final[int] = 100000
#: Default inactivity timeout for :meth:`ConnectionTracker.cleanup_stale`.
DEFAULT_STALE_TIMEOUT_SECONDS: Final[int] = 300
#: How many domains ``GlobalConnectionTable`` reports.
_TOP_DOMAIN_LIMIT: Final[int] = 20


@dataclass(frozen=True, slots=True)
class TrackerStats:
    """Per-FP tracker counters.  Mirrors ``ConnectionTracker::TrackerStats``."""

    active_connections: int = 0
    total_connections_seen: int = 0
    classified_connections: int = 0
    blocked_connections: int = 0


@dataclass(slots=True)
class GlobalStats:
    """Aggregated view across FPs.  Mirrors ``GlobalConnectionTable::GlobalStats``."""

    total_active_connections: int = 0
    total_connections_seen: int = 0
    app_distribution: dict[AppType, int] = field(default_factory=dict)
    top_domains: list[tuple[str, int]] = field(default_factory=list)


# ============================================================================
# Connection Tracker - flow table for one FP thread
# ============================================================================
class ConnectionTracker:
    """Maintains the flow table for a single FP thread.

    Mirrors ``class ConnectionTracker``.  Deliberately unsynchronised, as in
    the original: the owning FP thread is the only writer.
    """

    __slots__ = (
        "_fp_id",
        "_max_connections",
        "_connections",
        "_total_seen",
        "_classified_count",
        "_blocked_count",
    )

    def __init__(self, fp_id: int, max_connections: int = DEFAULT_MAX_CONNECTIONS) -> None:
        self._fp_id = fp_id
        self._max_connections = max_connections

        # Connection table.  FiveTuple hashing ensures consistent mapping, so
        # bidirectional flows are not handled specially here -- see
        # get_or_create_connection() for what that implies.
        self._connections: dict[FiveTuple, Connection] = {}

        # Statistics
        self._total_seen = 0
        self._classified_count = 0
        self._blocked_count = 0

    # ------------------------------------------------------------------
    # Lookup / creation
    # ------------------------------------------------------------------
    def get_or_create_connection(self, tuple_: FiveTuple) -> Connection:
        """Return the entry for ``tuple_``'s flow, creating one if absent.

        FIXED (was UPSTREAM BUG).  The C++ looked up only the exact tuple, even
        though :meth:`get_connection` right below it already knew to try the
        reverse.  Since the fast path calls *this* method, every conversation
        occupied two independent entries: the client->server direction and the
        server->client direction were tracked, classified and counted
        separately.  Three consequences, all now fixed:

        * "Active Connections" in every report was really a count of
          unidirectional flows, roughly double the true figure.
        * ``ConnectionState.ESTABLISHED`` was unreachable, because the
          handshake test needs ``syn_seen`` (client side) and ``syn_ack_seen``
          (server side) on the *same* record.
        * A flow classified from the client's Client Hello learned nothing from
          the server's replies, and vice versa.

        Both directions now share one record, keyed on whichever tuple was seen
        first.  Use :meth:`is_outbound` to ask which way a given packet ran.
        """
        existing = self._connections.get(tuple_)
        if existing is not None:
            return existing

        # Match the reverse direction of an already-tracked conversation.
        reverse = self._connections.get(tuple_.reverse())
        if reverse is not None:
            return reverse

        # Check if we need to evict old connections
        if len(self._connections) >= self._max_connections:
            self._evict_oldest()

        # Create new connection
        now = time.monotonic()
        conn = Connection(
            tuple=tuple_,
            state=ConnectionState.NEW,
            first_seen=now,
            last_seen=now,
        )

        self._connections[tuple_] = conn
        self._total_seen += 1

        return conn

    @staticmethod
    def is_outbound(conn: Connection, tuple_: FiveTuple) -> bool:
        """Return whether ``tuple_`` runs in the connection's original direction.

        A connection records the five-tuple of the first packet that created
        it, so a packet whose tuple matches exactly is "outbound" (same
        direction as the opener) and anything else is the return path.  No C++
        counterpart — it did not distinguish the two.
        """
        return conn.tuple == tuple_

    def get_connection(self, tuple_: FiveTuple) -> Connection | None:
        """Return an existing entry, trying the reverse tuple too.

        Mirrors ``getConnection``.  Returns ``None`` (C++ ``nullptr``) if
        neither direction is tracked.

        Note this method is unused by the fast path in the original.
        """
        found = self._connections.get(tuple_)
        if found is not None:
            return found

        # Try reverse tuple (for bidirectional matching)
        return self._connections.get(tuple_.reverse())

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def update_connection(
        self, conn: Connection | None, packet_size: int, is_outbound: bool
    ) -> None:
        """Record a packet against a connection.  Mirrors ``updateConnection``."""
        if conn is None:
            return

        conn.last_seen = time.monotonic()

        if is_outbound:
            conn.packets_out += 1
            conn.bytes_out += packet_size
        else:
            conn.packets_in += 1
            conn.bytes_in += packet_size

    def classify_connection(self, conn: Connection | None, app: AppType, sni: str) -> None:
        """Record a classification result.  Mirrors ``classifyConnection``.

        Classification is **sticky**: once a connection reaches
        ``CLASSIFIED`` it is never reclassified, so the first identifiable
        payload wins for the life of the flow.
        """
        if conn is None:
            return

        if conn.state != ConnectionState.CLASSIFIED:
            conn.app_type = app
            conn.sni = sni
            conn.state = ConnectionState.CLASSIFIED
            self._classified_count += 1

    def block_connection(self, conn: Connection | None) -> None:
        """Mark a connection blocked.  Mirrors ``blockConnection``.

        NOTE: ``blocked_count`` increments on **every call**, not once per
        connection, so it counts block *events*.  The fast path returns DROP
        early for an already-blocked connection, so in practice this fires once
        per flow — but the counter's meaning is events, and that is preserved.
        """
        if conn is None:
            return

        conn.state = ConnectionState.BLOCKED
        conn.action = PacketAction.DROP
        self._blocked_count += 1

    def close_connection(self, tuple_: FiveTuple) -> None:
        """Mark a connection closed.  Mirrors ``closeConnection``.

        Only the exact tuple is considered; no reverse lookup.
        """
        conn = self._connections.get(tuple_)
        if conn is not None:
            conn.state = ConnectionState.CLOSED

    def cleanup_stale(self, timeout_seconds: float = DEFAULT_STALE_TIMEOUT_SECONDS) -> int:
        """Remove timed-out and closed connections; return how many went.

        Mirrors ``cleanupStale``.

        The age comparison reproduces
        ``duration_cast<std::chrono::seconds>(now - last_seen) > timeout``,
        which **truncates to whole seconds** before comparing — so an entry
        idle for 300.9s is *not* stale against a 300s timeout.  Hence the
        ``int()`` below rather than a direct float comparison.
        """
        now = time.monotonic()
        removed = 0

        for key, conn in list(self._connections.items()):
            age_seconds = int(now - conn.last_seen)  # duration_cast truncation

            if age_seconds > timeout_seconds or conn.state == ConnectionState.CLOSED:
                del self._connections[key]
                removed += 1

        return removed

    def clear(self) -> None:
        """Drop every connection.  Mirrors ``clear``.

        Note the cumulative counters (``total_seen`` and friends) are *not*
        reset, matching the original.
        """
        self._connections.clear()

    def _evict_oldest(self) -> None:
        """Evict the least-recently-seen connection.  Mirrors ``evictOldest``.

        A linear scan over the whole table, as in the original — O(n) per
        eviction, so a full table degrades sharply.  Preserved.
        """
        if not self._connections:
            return

        oldest_key = min(
            self._connections,
            key=lambda k: self._connections[k].last_seen,
        )
        del self._connections[oldest_key]

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def get_all_connections(self) -> list[Connection]:
        """Return every tracked connection.  Mirrors ``getAllConnections``.

        The C++ returns copies; this returns the live objects.  Callers here
        only read them, and copying every ``Connection`` would be a needless
        expense — but treat the result as read-only.
        """
        return list(self._connections.values())

    def get_active_count(self) -> int:
        """Return the number of tracked connections.  Mirrors ``getActiveCount``."""
        return len(self._connections)

    def get_stats(self) -> TrackerStats:
        """Return this tracker's counters.  Mirrors ``getStats``."""
        return TrackerStats(
            active_connections=len(self._connections),
            total_connections_seen=self._total_seen,
            classified_connections=self._classified_count,
            blocked_connections=self._blocked_count,
        )

    def for_each(self, callback: Callable[[Connection], None]) -> None:
        """Invoke ``callback`` for every connection.  Mirrors ``forEach``.

        Iterates a snapshot — see the module docstring on why.
        """
        for conn in list(self._connections.values()):
            callback(conn)

    @property
    def fp_id(self) -> int:
        """The FP thread this tracker belongs to."""
        return self._fp_id

    def __len__(self) -> int:
        return len(self._connections)

    def __repr__(self) -> str:
        return f"ConnectionTracker(fp_id={self._fp_id}, active={len(self._connections)})"

    # C++-style aliases
    getOrCreateConnection = get_or_create_connection
    isOutbound = is_outbound
    getConnection = get_connection
    updateConnection = update_connection
    classifyConnection = classify_connection
    blockConnection = block_connection
    closeConnection = close_connection
    cleanupStale = cleanup_stale
    getAllConnections = get_all_connections
    getActiveCount = get_active_count
    getStats = get_stats
    forEach = for_each


# ============================================================================
# Global Connection Table - aggregates stats from all FP trackers
# ============================================================================
class GlobalConnectionTable:
    """Aggregates statistics across every FP's tracker.

    Mirrors ``class GlobalConnectionTable``.
    """

    __slots__ = ("_trackers", "_lock")

    def __init__(self, num_fps: int) -> None:
        self._trackers: list[ConnectionTracker | None] = [None] * num_fps
        self._lock = threading.Lock()

    def register_tracker(self, fp_id: int, tracker: ConnectionTracker) -> None:
        """Register an FP's tracker.  Mirrors ``registerTracker``.

        An ``fp_id`` beyond the configured size is silently ignored, as in the
        original's ``if (fp_id < trackers_.size())`` guard.  Note a negative
        ``fp_id`` would index from the end in Python where C++ would write out
        of bounds; the explicit lower bound below avoids both.
        """
        with self._lock:
            if 0 <= fp_id < len(self._trackers):
                self._trackers[fp_id] = tracker

    def get_global_stats(self) -> GlobalStats:
        """Aggregate counters, app distribution and top domains.

        Mirrors ``getGlobalStats``.
        """
        with self._lock:
            trackers = [t for t in self._trackers if t is not None]

        stats = GlobalStats()
        domain_counts: dict[str, int] = {}

        for tracker in trackers:
            tracker_stats = tracker.get_stats()
            stats.total_active_connections += tracker_stats.active_connections
            stats.total_connections_seen += tracker_stats.total_connections_seen

            # Collect app distribution
            def collect(conn: Connection) -> None:
                stats.app_distribution[conn.app_type] = (
                    stats.app_distribution.get(conn.app_type, 0) + 1
                )
                if conn.sni:
                    domain_counts[conn.sni] = domain_counts.get(conn.sni, 0) + 1

            tracker.for_each(collect)

        # Get top domains, highest count first.
        #
        # The C++ uses std::sort, which is UNSTABLE, over an unordered_map, so
        # ties came out in an arbitrary and run-dependent order.  Python's
        # sorted() is stable over an insertion-ordered dict, making ties
        # deterministic (first-seen wins).  Counts are identical either way.
        domain_vec = sorted(domain_counts.items(), key=lambda kv: -kv[1])
        stats.top_domains = domain_vec[:_TOP_DOMAIN_LIMIT]

        return stats

    def generate_report(self) -> str:
        """Render the connection statistics report.

        Mirrors ``generateReport`` character for character, including the
        box-drawing frame and the ``setw``/``setprecision`` field widths.
        """
        stats = self.get_global_stats()

        lines: list[str] = []
        lines.append("\n╔══════════════════════════════════════════════════════════════╗")
        lines.append("║               CONNECTION STATISTICS REPORT                    ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        lines.append(
            f"║ Active Connections:     {stats.total_active_connections:>10}"
            "                          ║"
        )
        lines.append(
            f"║ Total Connections Seen: {stats.total_connections_seen:>10}"
            "                          ║"
        )

        lines.append("╠══════════════════════════════════════════════════════════════╣")
        lines.append("║                    APPLICATION BREAKDOWN                      ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Calculate total for percentages
        total = sum(stats.app_distribution.values())

        # Sort by count (see the stability note in get_global_stats)
        sorted_apps = sorted(stats.app_distribution.items(), key=lambda kv: -kv[1])

        for app, count in sorted_apps:
            pct = (100.0 * count / total) if total > 0 else 0.0
            lines.append(
                f"║ {app_type_to_string(app):<20}{count:>10}"
                f" ({pct:>5.1f}%)           ║"
            )

        if stats.top_domains:
            lines.append("╠══════════════════════════════════════════════════════════════╣")
            lines.append("║                      TOP DOMAINS                             ║")
            lines.append("╠══════════════════════════════════════════════════════════════╣")

            for domain, count in stats.top_domains:
                if len(domain) > 35:
                    domain = domain[:32] + "..."
                lines.append(f"║ {domain:<40}{count:>10}           ║")

        lines.append("╚══════════════════════════════════════════════════════════════╝")

        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:
        registered = sum(1 for t in self._trackers if t is not None)
        return f"GlobalConnectionTable({registered}/{len(self._trackers)} trackers)"

    # C++-style aliases
    registerTracker = register_tracker
    getGlobalStats = get_global_stats
    generateReport = generate_report


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    def tup(sp: int, dp: int, proto: int = 6) -> FiveTuple:
        return FiveTuple(0x0A01A8C0, 0x08080808, sp, dp, proto)

    tracker = ConnectionTracker(fp_id=0, max_connections=3)

    # Creation and identity: the returned object IS the stored one
    c1 = tracker.get_or_create_connection(tup(1000, 443))
    assert tracker.get_or_create_connection(tup(1000, 443)) is c1
    c1.bytes_out += 7
    assert tracker.get_all_connections()[0].bytes_out == 7

    # FIXED: the reverse direction now shares the same record
    c_rev = tracker.get_or_create_connection(tup(1000, 443).reverse())
    assert c_rev is c1, "both directions must share one Connection"
    assert tracker.get_active_count() == 1
    assert tracker.get_connection(tup(1000, 443).reverse()) is c1
    # ...and direction is recoverable
    assert ConnectionTracker.is_outbound(c1, tup(1000, 443))
    assert not ConnectionTracker.is_outbound(c1, tup(1000, 443).reverse())

    # Updates
    tracker.update_connection(c1, 100, is_outbound=True)
    tracker.update_connection(c1, 50, is_outbound=False)
    assert (c1.packets_out, c1.bytes_out, c1.packets_in, c1.bytes_in) == (1, 107, 1, 50)
    tracker.update_connection(None, 10, True)  # nullptr guard

    # Classification is sticky
    tracker.classify_connection(c1, AppType.YOUTUBE, "www.youtube.com")
    tracker.classify_connection(c1, AppType.NETFLIX, "www.netflix.com")
    assert c1.app_type is AppType.YOUTUBE and c1.state is ConnectionState.CLASSIFIED
    assert tracker.get_stats().classified_connections == 1

    # Blocking counts events
    tracker.block_connection(c1)
    tracker.block_connection(c1)
    assert c1.state is ConnectionState.BLOCKED and c1.action is PacketAction.DROP
    assert tracker.get_stats().blocked_connections == 2

    # LRU eviction at capacity
    tracker.get_or_create_connection(tup(1001, 443))
    tracker.get_or_create_connection(tup(1003, 443))
    assert tracker.get_active_count() == 3
    tracker.get_or_create_connection(tup(1002, 443))  # triggers evict
    assert tracker.get_active_count() == 3, tracker.get_active_count()

    # Stale cleanup: CLOSED goes immediately; fresh entries stay
    t2 = ConnectionTracker(fp_id=1)
    a = t2.get_or_create_connection(tup(2000, 80))
    b = t2.get_or_create_connection(tup(2001, 80))
    b.state = ConnectionState.CLOSED
    assert t2.cleanup_stale(300) == 1
    assert t2.get_active_count() == 1
    a.last_seen = time.monotonic() - 301
    assert t2.cleanup_stale(300) == 1
    # Truncation: 300.9s idle is NOT stale against a 300s timeout
    t3 = ConnectionTracker(fp_id=2)
    c = t3.get_or_create_connection(tup(3000, 80))
    c.last_seen = time.monotonic() - 300.9
    assert t3.cleanup_stale(300) == 0, "duration_cast truncates to whole seconds"

    # Global aggregation + report
    gct = GlobalConnectionTable(num_fps=2)
    gct.register_tracker(0, tracker)
    gct.register_tracker(1, t2)
    gct.register_tracker(99, t3)  # out of range -> ignored
    g = gct.get_global_stats()
    assert g.total_active_connections == tracker.get_active_count() + t2.get_active_count()
    print(gct.generate_report())
    print("connection_tracker.py self-test OK")
