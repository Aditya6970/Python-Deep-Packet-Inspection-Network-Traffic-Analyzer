"""Fast-path processor threads: the DPI workhorses.

Python port of ``include/fast_path.h`` + ``src/fast_path.cpp``
(C++ ``namespace DPI``).

Each FP thread:

1. Receives packets from its input queue (fed by an LB)
2. Tracks connection state
3. Inspects payloads (SNI / HTTP Host / DNS) to classify the flow
4. Matches blocking rules
5. Forwards or drops via the output callback

C++ concepts replaced
---------------------
``using PacketOutputCallback = std::function<void(const PacketJob&, PacketAction)>``
    Becomes ``Callable[[PacketJob, PacketAction], None]``.

``std::thread`` + ``std::atomic<bool>``
    Become :class:`threading.Thread` and :class:`threading.Event`.

``std::atomic<uint64_t>`` counters
    Become :class:`~dpi.types.AtomicCounter`.  These are read by the reporting
    thread while the FP thread increments them, so the lock is doing real work
    here, not just ceremony.

``Connection*`` (possibly null)
    Becomes ``Connection | None``.  ``getOrCreateConnection`` never actually
    returns null, but the C++ null check is preserved.

``const uint8_t* payload = job.data.data() + job.payload_offset``
    Becomes a :class:`memoryview` slice — the same zero-copy borrow.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Final

from .connection_tracker import ConnectionTracker
from .rule_manager import BlockReason, RuleManager
from .sni_extractor import DNSExtractor, HTTPHostExtractor, SNIExtractor
from .thread_safe_queue import ThreadSafeQueue
from .types import (
    AppType,
    Connection,
    ConnectionState,
    PacketAction,
    PacketJob,
    app_type_to_string,
    sni_to_app_type,
)
from .types import AtomicCounter

__all__ = [
    "FP_QUEUE_SIZE",
    "FP_POP_TIMEOUT_MS",
    "FP_CLEANUP_TIMEOUT_SECONDS",
    "FP_CLEANUP_INTERVAL_SECONDS",
    "HTTP_PORTS",
    "PacketOutputCallback",
    "FPStats",
    "AggregatedFPStats",
    "FastPathProcessor",
    "FPManager",
]

#: Input queue capacity, hard-coded in the C++ constructor's initialiser list.
FP_QUEUE_SIZE: Final[int] = 10000
#: Poll interval so the run loop can observe the running flag.
FP_POP_TIMEOUT_MS: Final[float] = 100.0
#: Idle timeout applied by the stale-connection sweep.
FP_CLEANUP_TIMEOUT_SECONDS: Final[int] = 300
#: How often the sweep runs, busy or idle.  No C++ counterpart.
FP_CLEANUP_INTERVAL_SECONDS: Final[float] = 30.0

#: Callback invoked with every processed packet and its verdict.
PacketOutputCallback = Callable[[PacketJob, PacketAction], None]

#: Ports treated as plaintext HTTP.  The C++ knew only 80.
HTTP_PORTS: Final[frozenset[int]] = frozenset({80, 8000, 8080, 8888})

# TCP flag masks, redeclared locally exactly as ``updateTCPState`` does.
_SYN: Final[int] = 0x02
_ACK: Final[int] = 0x10
_FIN: Final[int] = 0x01
_RST: Final[int] = 0x04


@dataclass(frozen=True, slots=True)
class FPStats:
    """Per-FP counters.  Mirrors ``FastPathProcessor::FPStats``."""

    packets_processed: int = 0
    packets_forwarded: int = 0
    packets_dropped: int = 0
    connections_tracked: int = 0
    sni_extractions: int = 0
    classification_hits: int = 0


@dataclass(frozen=True, slots=True)
class AggregatedFPStats:
    """Totals across FPs.  Mirrors ``FPManager::AggregatedStats``."""

    total_processed: int = 0
    total_forwarded: int = 0
    total_dropped: int = 0
    total_connections: int = 0


# ============================================================================
# Fast Path Processor Thread
# ============================================================================
class FastPathProcessor:
    """One DPI worker thread.  Mirrors ``class FastPathProcessor``."""

    __slots__ = (
        "_fp_id",
        "_input_queue",
        "_conn_tracker",
        "_rule_manager",
        "_output_callback",
        "_packets_processed",
        "_packets_forwarded",
        "_packets_dropped",
        "_sni_extractions",
        "_classification_hits",
        "_running",
        "_thread",
        "_last_cleanup",
    )

    def __init__(
        self,
        fp_id: int,
        rule_manager: RuleManager | None,
        output_callback: PacketOutputCallback | None,
    ) -> None:
        self._fp_id = fp_id

        # Input queue from LB
        self._input_queue: ThreadSafeQueue[PacketJob] = ThreadSafeQueue(FP_QUEUE_SIZE)

        # Connection tracker (per-FP, no sharing needed)
        self._conn_tracker = ConnectionTracker(fp_id)

        # Rule manager (shared, read-only from here)
        self._rule_manager = rule_manager

        # Output callback
        self._output_callback = output_callback

        # Statistics
        self._packets_processed = AtomicCounter()
        self._packets_forwarded = AtomicCounter()
        self._packets_dropped = AtomicCounter()
        self._sni_extractions = AtomicCounter()
        self._classification_hits = AtomicCounter()

        # Thread control
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_cleanup = time.monotonic()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the FP thread.  Mirrors ``start()``; a no-op if already running."""
        if self._running.is_set():
            return

        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name=f"FP{self._fp_id}", daemon=True
        )
        self._thread.start()

        print(f"[FP{self._fp_id}] Started")

    def stop(self) -> None:
        """Stop the FP thread and join it.  Mirrors ``stop()``.

        NOTE: this clears the running flag and shuts down the queue, so the run
        loop exits as soon as its current ``pop_with_timeout`` returns — it
        does **not** drain whatever is still queued.  Packets left in the queue
        at stop time are never processed, and because
        :meth:`~dpi.thread_safe_queue.ThreadSafeQueue.push` silently discards
        after shutdown, no one is told.  Preserved; see
        :class:`~dpi.dpi_engine.DPIEngine` for how the engine avoids it by
        waiting for the queues to empty first.
        """
        if not self._running.is_set():
            return

        self._running.clear()
        self._input_queue.shutdown()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        self._thread = None

        print(
            f"[FP{self._fp_id}] Stopped (processed "
            f"{self._packets_processed.get()} packets)"
        )

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------
    def _run(self) -> None:
        """Pop, process, emit.  Mirrors ``FastPathProcessor::run``."""
        while self._running.is_set():
            # Get packet from input queue
            job = self._input_queue.pop_with_timeout(FP_POP_TIMEOUT_MS)

            # FIXED (was UPSTREAM BUG): the C++ swept stale connections only
            # in the queue-empty branch, so a continuously busy FP -- exactly
            # the case where the table grows -- never swept at all, and fell
            # back on O(n) LRU eviction once full.  The sweep is now driven by
            # elapsed time, so it happens whether or not the FP is idle.
            now = time.monotonic()
            if now - self._last_cleanup >= FP_CLEANUP_INTERVAL_SECONDS:
                self._last_cleanup = now
                self._conn_tracker.cleanup_stale(FP_CLEANUP_TIMEOUT_SECONDS)

            if job is None:
                continue

            self._packets_processed.increment()

            # Process the packet
            action = self._process_packet(job)

            # Call output callback
            if self._output_callback is not None:
                self._output_callback(job, action)

            # Update stats.  Anything that is not DROP counts as forwarded,
            # so INSPECT and LOG_ONLY would land in the forwarded bucket.
            if action == PacketAction.DROP:
                self._packets_dropped.increment()
            else:
                self._packets_forwarded.increment()

    # ------------------------------------------------------------------
    # Packet processing
    # ------------------------------------------------------------------
    def _process_packet(self, job: PacketJob) -> PacketAction:
        """Track, inspect and adjudicate one packet.  Mirrors ``processPacket``."""
        # Get or create connection
        conn = self._conn_tracker.get_or_create_connection(job.tuple)
        if conn is None:
            # Should not happen, but handle gracefully
            return PacketAction.FORWARD

        # Update connection stats.
        # FIXED (was UPSTREAM BUG): the C++ hard-coded `is_outbound = true`
        # with the comment "in this model, all packets from user are
        # outbound", so packets_in and bytes_in were permanently zero in every
        # report and the return path of every conversation was invisible.
        # Now that both directions share one Connection, the real direction is
        # recoverable from whichever tuple opened the flow.
        is_outbound = ConnectionTracker.is_outbound(conn, job.tuple)
        self._conn_tracker.update_connection(conn, len(job.data), is_outbound)

        # Update TCP state if applicable
        if job.tuple.protocol == 6:  # TCP
            self._update_tcp_state(conn, job.tcp_flags)

        # If connection is already blocked, drop immediately
        if conn.state == ConnectionState.BLOCKED:
            return PacketAction.DROP

        # If connection not yet classified, try to inspect payload
        if conn.state != ConnectionState.CLASSIFIED and job.payload_length > 0:
            self._inspect_payload(job, conn)

        # Check rules (even for classified connections, as rules might change)
        return self._check_rules(job, conn)

    def _inspect_payload(self, job: PacketJob, conn: Connection) -> None:
        """Classify a flow from its payload.  Mirrors ``inspectPayload``.

        Order: TLS SNI, then HTTP Host, then DNS, then a port-based fallback.
        The first that succeeds wins.
        """
        if job.payload_length == 0 or job.payload_offset >= len(job.data):
            return

        payload = memoryview(job.data)[job.payload_offset :]

        # Try TLS SNI extraction first (most common for HTTPS)
        if self._try_extract_sni(job, conn):
            return

        # Try HTTP Host header extraction
        if self._try_extract_http_host(job, conn):
            return

        # Check for DNS (port 53)
        if job.tuple.dst_port == 53 or job.tuple.src_port == 53:
            domain = DNSExtractor.extract_query(payload, job.payload_length)
            if domain is not None:
                self._conn_tracker.classify_connection(conn, AppType.DNS, domain)
                return

        # Basic port-based classification as fallback
        if job.tuple.dst_port in HTTP_PORTS:
            self._conn_tracker.classify_connection(conn, AppType.HTTP, "")
        elif job.tuple.dst_port == 443:
            self._conn_tracker.classify_connection(conn, AppType.HTTPS, "")

    def _try_extract_sni(self, job: PacketJob, conn: Connection) -> bool:
        """Try TLS SNI extraction.  Mirrors ``tryExtractSNI``.

        The gate is ``dst_port != 443 && payload_length < 50`` — note the
        ``&&``, so inspection proceeds when the port is 443 **or** the payload
        is at least 50 bytes.  A large payload on any port is therefore
        speculatively parsed as TLS.  Preserved.
        """
        # Only for port 443 (HTTPS) or if it looks like TLS
        if job.tuple.dst_port != 443 and job.payload_length < 50:
            return False

        if job.payload_offset >= len(job.data) or job.payload_length == 0:
            return False

        payload = memoryview(job.data)[job.payload_offset :]
        sni = SNIExtractor.extract(payload, job.payload_length)
        if sni is not None:
            self._sni_extractions.increment()

            # Map SNI to app type
            app = sni_to_app_type(sni)
            self._conn_tracker.classify_connection(conn, app, sni)

            # A "hit" means we resolved something more specific than the
            # generic HTTPS fallback.
            if app != AppType.UNKNOWN and app != AppType.HTTPS:
                self._classification_hits.increment()

            return True

        return False

    def _try_extract_http_host(self, job: PacketJob, conn: Connection) -> bool:
        """Try HTTP Host extraction.  Mirrors ``tryExtractHTTPHost``.

        FIXED (was UPSTREAM BUG): the C++ required ``dst_port == 80`` exactly,
        so plaintext HTTP on 8080/8000/8888 was never inspected and — because
        the port-based fallback also only knew 80 and 443 — such flows stayed
        UNKNOWN forever.  Common alternate HTTP ports are now included; the
        extractor still verifies an HTTP method prefix before parsing, so a
        non-HTTP service on those ports is rejected as before.
        """
        if job.tuple.dst_port not in HTTP_PORTS:
            return False

        if job.payload_offset >= len(job.data) or job.payload_length == 0:
            return False

        payload = memoryview(job.data)[job.payload_offset :]
        host = HTTPHostExtractor.extract(payload, job.payload_length)
        if host is not None:
            app = sni_to_app_type(host)
            self._conn_tracker.classify_connection(conn, app, host)

            if app != AppType.UNKNOWN and app != AppType.HTTP:
                self._classification_hits.increment()

            return True

        return False

    def _check_rules(self, job: PacketJob, conn: Connection) -> PacketAction:
        """Apply blocking rules.  Mirrors ``checkRules``."""
        if self._rule_manager is None:
            return PacketAction.FORWARD

        # Parse source IP from tuple
        src_ip = job.tuple.src_ip

        # Check blocking rules
        block_reason = self._rule_manager.should_block(
            src_ip,
            job.tuple.dst_port,
            conn.app_type,
            conn.sni,
        )

        if block_reason is not None:
            # Log the block.  One line PER PACKET, not per connection -- a
            # blocked bulk transfer produces a line for every packet until the
            # connection is marked BLOCKED and short-circuits above.
            label = {
                BlockReason.Type.IP: "IP",
                BlockReason.Type.APP: "App",
                BlockReason.Type.DOMAIN: "Domain",
                BlockReason.Type.PORT: "Port",
            }[block_reason.type]

            print(f"[FP{self._fp_id}] BLOCKED packet: {label} {block_reason.detail}")

            # Mark connection as blocked
            self._conn_tracker.block_connection(conn)

            return PacketAction.DROP

        return PacketAction.FORWARD

    @staticmethod
    def _update_tcp_state(conn: Connection, tcp_flags: int) -> None:
        """Advance the TCP state machine.  Mirrors ``updateTCPState``.

        The handshake test requires ``syn_seen`` and ``syn_ack_seen`` on the
        *same* Connection.  In the C++ that was unreachable, because each
        direction of a conversation had its own record; now that both
        directions share one (see
        :meth:`~dpi.connection_tracker.ConnectionTracker.get_or_create_connection`),
        a normal three-way handshake really does reach ``ESTABLISHED``.
        """
        if tcp_flags & _SYN:
            if tcp_flags & _ACK:
                conn.syn_ack_seen = True
            else:
                conn.syn_seen = True

        if conn.syn_seen and conn.syn_ack_seen and (tcp_flags & _ACK):
            if conn.state == ConnectionState.NEW:
                conn.state = ConnectionState.ESTABLISHED

        if tcp_flags & _FIN:
            conn.fin_seen = True

        if tcp_flags & _RST:
            conn.state = ConnectionState.CLOSED

        if conn.fin_seen and (tcp_flags & _ACK):
            conn.state = ConnectionState.CLOSED

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_input_queue(self) -> ThreadSafeQueue[PacketJob]:
        """Return the input queue for the LB to push into."""
        return self._input_queue

    def get_connection_tracker(self) -> ConnectionTracker:
        """Return this FP's connection tracker.  Mirrors ``getConnectionTracker``."""
        return self._conn_tracker

    def get_stats(self) -> FPStats:
        """Return this FP's counters.  Mirrors ``getStats``."""
        return FPStats(
            packets_processed=self._packets_processed.get(),
            packets_forwarded=self._packets_forwarded.get(),
            packets_dropped=self._packets_dropped.get(),
            connections_tracked=self._conn_tracker.get_active_count(),
            sni_extractions=self._sni_extractions.get(),
            classification_hits=self._classification_hits.get(),
        )

    def get_id(self) -> int:
        """Return this FP's id.  Mirrors ``getId``."""
        return self._fp_id

    def is_running(self) -> bool:
        """Return whether the thread is running.  Mirrors ``isRunning``."""
        return self._running.is_set()

    def __repr__(self) -> str:
        return (
            f"FastPathProcessor(id={self._fp_id}, running={self.is_running()}, "
            f"processed={self._packets_processed.get()})"
        )

    # C++-style aliases
    getInputQueue = get_input_queue
    getConnectionTracker = get_connection_tracker
    getStats = get_stats
    getId = get_id
    isRunning = is_running


# ============================================================================
# FP Manager - creates and manages multiple FP threads
# ============================================================================
class FPManager:
    """Creates and supervises a pool of :class:`FastPathProcessor` threads.

    Mirrors ``class FPManager``.
    """

    __slots__ = ("_fps",)

    def __init__(
        self,
        num_fps: int,
        rule_manager: RuleManager | None,
        output_callback: PacketOutputCallback | None,
    ) -> None:
        # Create FP processors (each has its own input queue)
        self._fps = [
            FastPathProcessor(i, rule_manager, output_callback) for i in range(num_fps)
        ]

        print(f"[FPManager] Created {num_fps} fast path processors")

    def start_all(self) -> None:
        """Start every FP thread.  Mirrors ``startAll``."""
        for fp in self._fps:
            fp.start()

    def stop_all(self) -> None:
        """Stop every FP thread.  Mirrors ``stopAll``."""
        for fp in self._fps:
            fp.stop()

    def get_fp(self, fp_id: int) -> FastPathProcessor:
        """Return a specific FP.  Mirrors ``getFP``."""
        return self._fps[fp_id]

    def get_fp_queue(self, fp_id: int) -> ThreadSafeQueue[PacketJob]:
        """Return an FP's input queue.  Mirrors ``getFPQueue``."""
        return self._fps[fp_id].get_input_queue()

    def get_queue_ptrs(self) -> list[ThreadSafeQueue[PacketJob]]:
        """Return every FP input queue, for the LB manager.

        Mirrors ``getQueuePtrs``.
        """
        return [fp.get_input_queue() for fp in self._fps]

    def get_num_fps(self) -> int:
        """Return the number of FP threads.  Mirrors ``getNumFPs``."""
        return len(self._fps)

    def get_aggregated_stats(self) -> AggregatedFPStats:
        """Sum counters across FPs.  Mirrors ``getAggregatedStats``."""
        total_processed = total_forwarded = total_dropped = total_connections = 0
        for fp in self._fps:
            s = fp.get_stats()
            total_processed += s.packets_processed
            total_forwarded += s.packets_forwarded
            total_dropped += s.packets_dropped
            total_connections += s.connections_tracked
        return AggregatedFPStats(
            total_processed=total_processed,
            total_forwarded=total_forwarded,
            total_dropped=total_dropped,
            total_connections=total_connections,
        )

    def generate_classification_report(self) -> str:
        """Render the application classification report.

        Mirrors ``generateClassificationReport`` character for character,
        including the ``#`` bar graph (one ``#`` per 5%, 20 wide).
        """
        # Aggregate app distribution across all FPs
        app_counts: dict[AppType, int] = {}
        domain_counts: dict[str, int] = {}
        total_classified = 0
        total_unknown = 0

        for fp in self._fps:

            def collect(conn: Connection) -> None:
                nonlocal total_classified, total_unknown
                app_counts[conn.app_type] = app_counts.get(conn.app_type, 0) + 1

                if conn.app_type == AppType.UNKNOWN:
                    total_unknown += 1
                else:
                    total_classified += 1

                if conn.sni:
                    domain_counts[conn.sni] = domain_counts.get(conn.sni, 0) + 1

            fp.get_connection_tracker().for_each(collect)

        lines: list[str] = []
        lines.append("\n╔══════════════════════════════════════════════════════════════╗")
        lines.append("║                 APPLICATION CLASSIFICATION REPORT             ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        total = total_classified + total_unknown
        classified_pct = (100.0 * total_classified / total) if total > 0 else 0.0
        unknown_pct = (100.0 * total_unknown / total) if total > 0 else 0.0

        lines.append(f"║ Total Connections:    {total:>10}                           ║")
        lines.append(
            f"║ Classified:           {total_classified:>10}"
            f" ({classified_pct:.1f}%)                  ║"
        )
        lines.append(
            f"║ Unidentified:         {total_unknown:>10}"
            f" ({unknown_pct:.1f}%)                  ║"
        )

        lines.append("╠══════════════════════════════════════════════════════════════╣")
        lines.append("║                    APPLICATION DISTRIBUTION                   ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Sort apps by count.  std::sort is unstable in C++, so ties came out
        # arbitrarily; sorted() is stable over an insertion-ordered dict, which
        # makes ties deterministic.  Counts are identical either way.
        sorted_apps = sorted(app_counts.items(), key=lambda kv: -kv[1])

        for app, count in sorted_apps:
            pct = (100.0 * count / total) if total > 0 else 0.0

            # Create a simple bar graph
            bar_len = int(pct / 5)  # 20 chars max
            bar = "#" * bar_len

            lines.append(
                f"║ {app_type_to_string(app):<15}{count:>8}"
                f" {pct:>5.1f}% {bar:<20}   ║"
            )

        lines.append("╚══════════════════════════════════════════════════════════════╝")

        return "\n".join(lines) + "\n"

    def __len__(self) -> int:
        return len(self._fps)

    def __repr__(self) -> str:
        s = self.get_aggregated_stats()
        return f"FPManager(fps={len(self._fps)}, processed={s.total_processed})"

    # C++-style aliases
    startAll = start_all
    stopAll = stop_all
    getFP = get_fp
    getFPQueue = get_fp_queue
    getQueuePtrs = get_queue_ptrs
    getNumFPs = get_num_fps
    getAggregatedStats = get_aggregated_stats
    generateClassificationReport = generate_classification_report


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import io
    import struct
    import time
    from contextlib import redirect_stdout

    from .types import FiveTuple

    def client_hello(host: bytes) -> bytes:
        sni = struct.pack(">H", len(host) + 3) + b"\x00" + struct.pack(">H", len(host)) + host
        exts = struct.pack(">HH", 0x0000, len(sni)) + sni
        body = (
            struct.pack(">H", 0x0303)
            + b"\xAB" * 32
            + b"\x00"
            + struct.pack(">H", 2)
            + b"\x13\x01"
            + b"\x01\x00"
            + struct.pack(">H", len(exts))
            + exts
        )
        hs = b"\x01" + struct.pack(">I", len(body))[1:] + body
        return b"\x16\x03\x03" + struct.pack(">H", len(hs)) + hs

    def make_job(pid: int, sport: int, dport: int, payload: bytes, flags: int = 0x18) -> PacketJob:
        header = b"\x00" * 54  # 14 eth + 20 ip + 20 tcp
        data = header + payload
        return PacketJob(
            packet_id=pid,
            tuple=FiveTuple(0x0A01A8C0, 0x08080808, sport, dport, 6),
            data=data,
            payload_offset=54,
            payload_length=len(payload),
            tcp_flags=flags,
        )

    outputs: list[tuple[int, PacketAction]] = []
    rules = RuleManager()
    quiet = io.StringIO()

    with redirect_stdout(quiet):
        rules.block_domain("*.tiktok.com")
        rules.block_port(8080)

        fp = FastPathProcessor(
            0, rules, lambda job, action: outputs.append((job.packet_id, action))
        )
        fp.start()

        jobs = [
            make_job(1, 5000, 443, client_hello(b"www.youtube.com")),
            make_job(2, 5001, 443, client_hello(b"cdn.tiktok.com")),      # blocked: domain
            # Same 5-tuple as #2: must short-circuit on state==BLOCKED,
            # so it is dropped WITHOUT a second SNI extraction.
            make_job(3, 5001, 443, client_hello(b"cdn.tiktok.com")),
            make_job(4, 5003, 80, b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"),
            make_job(5, 5004, 8080, b"GET / HTTP/1.1\r\nHost: a.com\r\n\r\n"),  # blocked: port
            make_job(6, 5005, 53, b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
                                  b"\x03www\x07example\x03com\x00\x00\x01\x00\x01"),
            make_job(7, 5006, 443, b""),  # no payload
        ]
        for j in jobs:
            fp.get_input_queue().push(j)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(outputs) < len(jobs):
            time.sleep(0.01)
        fp.stop()

        stats = fp.get_stats()
        conns = {c.tuple.src_port: c for c in fp.get_connection_tracker().get_all_connections()}

    verdicts = dict(outputs)
    assert verdicts[1] is PacketAction.FORWARD, verdicts
    assert verdicts[2] is PacketAction.DROP, verdicts
    assert verdicts[3] is PacketAction.DROP, verdicts
    assert verdicts[4] is PacketAction.FORWARD, verdicts
    assert verdicts[5] is PacketAction.DROP, verdicts
    assert verdicts[6] is PacketAction.FORWARD, verdicts

    assert conns[5000].app_type is AppType.YOUTUBE and conns[5000].sni == "www.youtube.com"
    assert conns[5001].app_type is AppType.TIKTOK
    assert conns[5003].sni == "example.com"
    assert conns[5005].app_type is AppType.DNS and conns[5005].sni == "www.example.com"
    assert conns[5006].app_type is AppType.UNKNOWN, "empty payload -> never classified"
    assert 5002 not in conns, "packet 3 reused flow 5001, so no 5002 entry exists"

    print(f"processed={stats.packets_processed} forwarded={stats.packets_forwarded} "
          f"dropped={stats.packets_dropped} sni={stats.sni_extractions} "
          f"hits={stats.classification_hits} conns={stats.connections_tracked}")
    assert stats.packets_processed == len(jobs)
    assert stats.packets_dropped == 3
    assert stats.sni_extractions == 2, stats
    assert stats.connections_tracked == 6, stats

    # Report rendering
    with redirect_stdout(quiet):
        mgr = FPManager(1, rules, None)
        t = mgr.get_fp(0).get_connection_tracker()
        for port, app, sni in (
            (1, AppType.YOUTUBE, "www.youtube.com"),
            (2, AppType.YOUTUBE, "www.youtube.com"),
            (3, AppType.GOOGLE, "www.google.com"),
            (4, AppType.UNKNOWN, ""),
        ):
            c = t.get_or_create_connection(FiveTuple(1, 2, port, 443, 6))
            if app is not AppType.UNKNOWN:
                t.classify_connection(c, app, sni)
    print(mgr.generate_classification_report())

    print("fast_path.py self-test OK")
