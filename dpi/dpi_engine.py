"""DPI Engine — the orchestrator.

Python port of ``include/dpi_engine.h`` + ``src/dpi_engine.cpp``
(C++ ``namespace DPI``).

Architecture::

    +------------------+
    |   PCAP Reader    |  (Reads packets from input file)
    +--------+---------+
             | hash to select LB
    +--------+----------+
    |   Load Balancers  |  (2 LB threads)
    +----+--------+-----+
         | hash to select FP within LB's pool
    +----+--------+-----+
    |  Fast Path Procs  |  (4 FP threads, 2 per LB)
    +----+--------+-----+
    +----+--------+-----+
    |   Output Queue    |  (Packets to forward)
    +--------+----------+
    |   Output Writer   |  (Writes to output PCAP)
    +-------------------+

C++ concepts replaced
---------------------
``std::unique_ptr<T>`` members
    Become ordinary attributes initialised to ``None``.  ``~DPIEngine()``
    calling ``stop()`` becomes a context manager plus an explicit
    :meth:`DPIEngine.stop`; Python has no deterministic destructor.

``std::ofstream output_file_`` + ``std::mutex output_mutex_``
    Become a binary file object guarded by a :class:`threading.Lock`.

Lambda output callback capturing ``this``
    Becomes a bound method reference — the same closure over engine state.

Fixed upstream bug
------------------
The C++ decided "processing is finished" with two fixed sleeps (500 ms then
200 ms) rather than by observing the queues.  On a capture large enough that
the pipeline was still draining when those elapsed, :meth:`stop` shut the
queues down and in-flight packets were discarded **silently** — the queue's
``push`` drops on shutdown without reporting
(see :meth:`~dpi.thread_safe_queue.ThreadSafeQueue.push`), so the forwarded
count simply came out short with no error.  :attr:`Config.drain_until_idle`
now defaults to ``True``: the engine waits for every queue to be observably
empty before stopping.  Set it to ``False`` to restore the original timing.
"""

from __future__ import annotations

import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Final

from .connection_tracker import GlobalConnectionTable
from .fast_path import FPManager
from .load_balancer import LBManager
from .packet_parser import PacketParser, ParsedPacket
from .pcap_reader import PcapGlobalHeader, PcapReader, RawPacket
from .rule_manager import RuleManager, RuleStats
from .thread_safe_queue import ThreadSafeQueue
from .types import (
    AppType,
    Connection,
    DPIStats,
    FiveTuple,
    PacketAction,
    PacketJob,
    app_type_to_string,
)

__all__ = ["Config", "DPIEngine", "FlowSnapshot"]

#: Output queue capacity, hard-coded in the C++ constructor's initialiser list.
OUTPUT_QUEUE_SIZE: Final[int] = 10000
#: Poll interval for the output writer thread.
OUTPUT_POP_TIMEOUT_MS: Final[float] = 100.0
#: Post-reader settle, from ``waitForCompletion``.
DRAIN_SLEEP_SECONDS: Final[float] = 0.5
#: Additional settle before ``stop()``, from ``processFile``.
FINAL_SLEEP_SECONDS: Final[float] = 0.2

_GLOBAL_HEADER_FMT: Final[str] = "=IHHiIII"
_PACKET_HEADER_FMT: Final[str] = "=IIII"


@dataclass(slots=True)
class Config:
    """Engine configuration.  Mirrors ``struct DPIEngine::Config``."""

    num_load_balancers: int = 2
    fps_per_lb: int = 2
    #: Present in the C++ struct but never read there; kept for parity.
    queue_size: int = 10000
    rules_file: str = ""
    #: Present in the C++ struct but never read there; kept for parity.
    verbose: bool = False

    #: FIXED (was UPSTREAM BUG): wait for the LB/FP/output queues to actually
    #: drain before shutting threads down, instead of trusting two fixed
    #: sleeps.  Set False to restore the original's timing-based behaviour,
    #: which silently discards in-flight packets on a large capture.
    drain_until_idle: bool = True


@dataclass(frozen=True, slots=True)
class FlowSnapshot:
    """An immutable, read-only view of one completed analysis run.

    Returned by :meth:`DPIEngine.get_flow_snapshot`.  Purely a container: it
    holds references to what the engine already computed and adds no logic of
    its own.

    This exists so an external consumer -- the optional ``ai/`` layer, a JSON
    exporter, a notebook -- can read results without reaching into the
    engine's private attributes.  The engine has no knowledge of who consumes
    it, and nothing in the packet path depends on this type.
    """

    #: Every tracked connection, gathered across all FP threads.
    connections: tuple[Connection, ...] = ()
    #: Capture-wide packet counters (see :meth:`DPIStats.snapshot`).
    packet_stats: dict[str, int] = field(default_factory=dict)
    #: Flow count per application.
    app_distribution: dict[AppType, int] = field(default_factory=dict)
    #: Most frequently seen server names, highest first.
    top_domains: tuple[tuple[str, int], ...] = ()
    #: Configured blocking-rule counts, or None if no rule manager exists.
    rule_stats: RuleStats | None = None
    num_load_balancers: int = 0
    fps_per_lb: int = 0


class DPIEngine:
    """Top-level engine: reader, LB pool, FP pool, output writer, reporting.

    Mirrors ``class DPIEngine``.
    """

    __slots__ = (
        "_config",
        "_rule_manager",
        "_global_conn_table",
        "_fp_manager",
        "_lb_manager",
        "_output_queue",
        "_output_thread",
        "_output_file",
        "_output_lock",
        "_stats",
        "_running",
        "_processing_complete",
        "_reader_thread",
    )

    def __init__(self, config: Config) -> None:
        self._config = config

        self._rule_manager: RuleManager | None = None
        self._global_conn_table: GlobalConnectionTable | None = None
        self._fp_manager: FPManager | None = None
        self._lb_manager: LBManager | None = None

        self._output_queue: ThreadSafeQueue[PacketJob] = ThreadSafeQueue(OUTPUT_QUEUE_SIZE)
        self._output_thread: threading.Thread | None = None
        self._output_file: BinaryIO | None = None
        self._output_lock = threading.Lock()

        self._stats = DPIStats()

        self._running = threading.Event()
        self._processing_complete = threading.Event()
        self._reader_thread: threading.Thread | None = None

        total = config.num_load_balancers * config.fps_per_lb
        print("")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║                    DPI ENGINE v1.0                            ║")
        print("║               Deep Packet Inspection System                   ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print("║ Configuration:                                                ║")
        print(f"║   Load Balancers:    {config.num_load_balancers:>3}                                       ║")
        print(f"║   FPs per LB:        {config.fps_per_lb:>3}                                       ║")
        print(f"║   Total FP threads:  {total:>3}                                       ║")
        print("╚══════════════════════════════════════════════════════════════╝")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        """Create the rule manager, thread pools and connection table.

        Mirrors ``initialize()``; always returns ``True``, as the original does.
        """
        # Create rule manager
        self._rule_manager = RuleManager()

        # Load rules if specified
        if self._config.rules_file:
            self._rule_manager.load_rules(self._config.rules_file)

        # Create FP manager (creates FP threads and their queues)
        total_fps = self._config.num_load_balancers * self._config.fps_per_lb
        self._fp_manager = FPManager(total_fps, self._rule_manager, self._handle_output)

        # Create LB manager (creates LB threads, connects to FP queues)
        self._lb_manager = LBManager(
            self._config.num_load_balancers,
            self._config.fps_per_lb,
            self._fp_manager.get_queue_ptrs(),
        )

        # Create global connection table
        self._global_conn_table = GlobalConnectionTable(total_fps)
        for i in range(total_fps):
            self._global_conn_table.register_tracker(
                i, self._fp_manager.get_fp(i).get_connection_tracker()
            )

        print("[DPIEngine] Initialized successfully")
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the output, FP and LB threads.  Mirrors ``start()``."""
        if self._running.is_set():
            return

        self._running.set()
        self._processing_complete.clear()

        # Start output thread
        self._output_thread = threading.Thread(
            target=self._output_thread_func, name="Output", daemon=True
        )
        self._output_thread.start()

        # Start FP threads
        assert self._fp_manager is not None
        self._fp_manager.start_all()

        # Start LB threads
        assert self._lb_manager is not None
        self._lb_manager.start_all()

        print("[DPIEngine] All threads started")

    def stop(self) -> None:
        """Stop every thread, LBs first.  Mirrors ``stop()``.

        Order matters and is preserved: LBs stop before FPs because LBs feed
        FPs, and the output thread stops last so it can flush what the FPs
        already emitted.
        """
        if not self._running.is_set():
            return

        self._running.clear()

        # Stop LB threads first (they feed FPs)
        if self._lb_manager is not None:
            self._lb_manager.stop_all()

        # Stop FP threads
        if self._fp_manager is not None:
            self._fp_manager.stop_all()

        # Stop output thread
        self._output_queue.shutdown()
        if self._output_thread is not None and self._output_thread.is_alive():
            self._output_thread.join()
        self._output_thread = None

        print("[DPIEngine] All threads stopped")

    def wait_for_completion(self) -> None:
        """Join the reader, then let the queues settle.

        Mirrors ``waitForCompletion()``.  The fixed sleep is the original's
        entire drain strategy; :attr:`Config.drain_until_idle` replaces it with
        a real check when enabled.
        """
        # Wait for reader to finish
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join()
        self._reader_thread = None

        if self._config.drain_until_idle:
            self._wait_until_queues_idle()
        else:
            # Wait a bit for queues to drain
            time.sleep(DRAIN_SLEEP_SECONDS)

        # Signal completion
        self._processing_complete.set()

    def _wait_until_queues_idle(self, timeout_seconds: float = 60.0) -> bool:
        """Block until every queue is empty (opt-in; no C++ counterpart).

        Returns ``True`` if the pipeline went idle, ``False`` on timeout.
        """
        assert self._lb_manager is not None and self._fp_manager is not None

        deadline = time.monotonic() + timeout_seconds
        stable = 0
        while time.monotonic() < deadline:
            depth = self._output_queue.size()
            for i in range(self._lb_manager.get_num_lbs()):
                depth += self._lb_manager.get_lb(i).get_input_queue().size()
            for i in range(self._fp_manager.get_num_fps()):
                depth += self._fp_manager.get_fp_queue(i).size()

            # Require several consecutive idle observations, so a packet in
            # flight *between* two queues is not mistaken for an empty pipeline.
            stable = stable + 1 if depth == 0 else 0
            if stable >= 5:
                return True
            time.sleep(0.02)

        return False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def process_file(self, input_file: str | Path, output_file: str | Path) -> bool:
        """Run a capture through the pipeline.  Mirrors ``processFile``."""
        print(f"\n[DPIEngine] Processing: {input_file}")
        print(f"[DPIEngine] Output to:  {output_file}\n")

        # Initialize if not already done
        if self._rule_manager is None:
            if not self.initialize():
                return False

        # Open output file
        try:
            self._output_file = Path(output_file).open("wb")
        except OSError:
            print("[DPIEngine] Error: Cannot open output file", file=sys.stderr)
            return False

        # Start processing threads
        self.start()

        # Start reader thread
        self._reader_thread = threading.Thread(
            target=self._reader_thread_func, args=(str(input_file),), name="Reader", daemon=True
        )
        self._reader_thread.start()

        # Wait for completion
        self.wait_for_completion()

        # Give some time for final packets to process
        if not self._config.drain_until_idle:
            time.sleep(FINAL_SLEEP_SECONDS)

        # Stop all threads
        self.stop()

        # Close output file
        with self._output_lock:
            if self._output_file is not None:
                self._output_file.close()
                self._output_file = None

        # Print final report
        sys.stdout.write(self.generate_report())
        assert self._fp_manager is not None
        sys.stdout.write(self._fp_manager.generate_classification_report())

        return True

    # ------------------------------------------------------------------
    # Reader
    # ------------------------------------------------------------------
    def _reader_thread_func(self, input_file: str) -> None:
        """Read, parse, filter and dispatch packets.  Mirrors ``readerThreadFunc``."""
        reader = PcapReader()

        if not reader.open(input_file):
            print("[Reader] Error: Cannot open input file", file=sys.stderr)
            return

        # Write PCAP header to output
        self._write_output_header(reader.get_global_header())

        packet_id = 0

        print("[Reader] Starting packet processing...")

        assert self._lb_manager is not None

        while True:
            raw = reader.read_next_packet()
            if raw is None:
                break

            # Parse the packet
            parsed = PacketParser.parse(raw)
            if parsed is None:
                continue  # Skip unparseable packets

            # Only process IP packets with TCP/UDP
            if not parsed.has_ip or (not parsed.has_tcp and not parsed.has_udp):
                continue

            # Create packet job
            job = self.create_packet_job(raw, parsed, packet_id)
            packet_id += 1

            # Update global stats
            self._stats.total_packets.increment()
            self._stats.total_bytes.add(len(raw.data))

            if parsed.has_tcp:
                self._stats.tcp_packets.increment()
            elif parsed.has_udp:
                self._stats.udp_packets.increment()

            # Send to appropriate LB based on hash
            lb = self._lb_manager.get_lb_for_packet(job.tuple)
            lb.get_input_queue().push(job)

        print(f"[Reader] Finished reading {packet_id} packets")
        reader.close()

    @staticmethod
    def _parse_ip(ip: str) -> int:
        """Re-parse a dotted quad into the wire-order integer.

        Mirrors the ``parseIP`` lambda inside ``createPacketJob`` — a verbatim
        third copy of the same routine that appears in ``RuleManager`` and in
        ``main_working``/``dpi_mt``.

        Note the round trip here is lossless *in this direction*: the string
        came from ``PacketParser::ipToString``, so every octet is already in
        0-255 and none of the parser's leniency is exercised.  It is still
        pure waste — the parser had the integer and stringified it one call
        earlier.  Preserved rather than short-circuited, so five-tuples stay
        bit-identical.
        """
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
        result |= (octet << shift) & 0xFFFFFFFF
        return result & 0xFFFFFFFF

    def create_packet_job(
        self, raw: RawPacket, parsed: ParsedPacket, packet_id: int
    ) -> PacketJob:
        """Build a :class:`~dpi.types.PacketJob`.  Mirrors ``createPacketJob``.

        Offsets are recomputed from the raw bytes rather than taken from
        ``parsed``, duplicating the parser's IHL and data-offset arithmetic.
        Preserved, including that duplication.
        """
        job = PacketJob()
        job.packet_id = packet_id
        job.ts_sec = raw.header.ts_sec
        job.ts_usec = raw.header.ts_usec

        # Set five-tuple - parse IP addresses from string back to uint32
        job.tuple = FiveTuple(
            src_ip=self._parse_ip(parsed.src_ip),
            dst_ip=self._parse_ip(parsed.dest_ip),
            src_port=parsed.src_port,
            dst_port=parsed.dest_port,
            protocol=parsed.protocol,
        )

        # TCP flags
        job.tcp_flags = parsed.tcp_flags

        # Copy packet data
        job.data = raw.data

        # Calculate offsets
        job.eth_offset = 0
        job.ip_offset = 14  # Ethernet header is 14 bytes

        # IP header length
        if len(job.data) > 14:
            ip_ihl = job.data[14] & 0x0F
            ip_header_len = ip_ihl * 4
            job.transport_offset = 14 + ip_header_len

            # Transport header length
            if parsed.has_tcp and len(job.data) > job.transport_offset:
                tcp_data_offset = (job.data[job.transport_offset + 12] >> 4) & 0x0F
                tcp_header_len = tcp_data_offset * 4
                job.payload_offset = job.transport_offset + tcp_header_len
            elif parsed.has_udp:
                job.payload_offset = job.transport_offset + 8  # UDP header is 8 bytes

            if job.payload_offset < len(job.data):
                job.payload_length = len(job.data) - job.payload_offset
                job.payload_data = memoryview(job.data)[job.payload_offset :]

        return job

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _output_thread_func(self) -> None:
        """Drain the output queue to the PCAP file.  Mirrors ``outputThreadFunc``."""
        while self._running.is_set() or not self._output_queue.empty():
            job = self._output_queue.pop_with_timeout(OUTPUT_POP_TIMEOUT_MS)

            if job is not None:
                self._write_output_packet(job)

    def _handle_output(self, job: PacketJob, action: PacketAction) -> None:
        """Route an FP verdict to the writer or the drop counter.

        Mirrors ``handleOutput``.  Called from every FP thread.
        """
        if action == PacketAction.DROP:
            self._stats.dropped_packets.increment()
            return

        self._stats.forwarded_packets.increment()
        self._output_queue.push(job)

    def _write_output_header(self, header: PcapGlobalHeader) -> bool:
        """Copy the input capture's global header.  Mirrors ``writeOutputHeader``.

        NOTE: the header is written back in **native** byte order.  For a
        byte-swapped input capture, ``PcapReader`` swapped some fields in
        memory but left ``magic_number``, ``thiszone`` and ``sigfigs`` alone,
        so the output file for such an input is malformed.  Preserved; it only
        affects captures whose endianness differs from the host.
        """
        with self._output_lock:
            if self._output_file is None:
                return False

            self._output_file.write(
                struct.pack(
                    _GLOBAL_HEADER_FMT,
                    header.magic_number,
                    header.version_major,
                    header.version_minor,
                    header.thiszone,
                    header.sigfigs,
                    header.snaplen,
                    header.network,
                )
            )
            return True

    def _write_output_packet(self, job: PacketJob) -> None:
        """Append one packet record.  Mirrors ``writeOutputPacket``.

        ``orig_len`` is set from the captured length, so a capture made with a
        snaplen shorter than the wire packet loses its true original length in
        the output.  Preserved.
        """
        with self._output_lock:
            if self._output_file is None:
                return

            size = len(job.data)
            self._output_file.write(
                struct.pack(_PACKET_HEADER_FMT, job.ts_sec, job.ts_usec, size, size)
            )
            self._output_file.write(job.data)

    # ------------------------------------------------------------------
    # Rule Management API
    # ------------------------------------------------------------------
    def block_ip(self, ip: str) -> None:
        """Block a source IP.  Mirrors ``blockIP``."""
        if self._rule_manager is not None:
            self._rule_manager.block_ip(ip)

    def unblock_ip(self, ip: str) -> None:
        """Unblock a source IP.  Mirrors ``unblockIP``."""
        if self._rule_manager is not None:
            self._rule_manager.unblock_ip(ip)

    def block_app(self, app: AppType | str) -> None:
        """Block an application, by enum or display name.  Mirrors both overloads.

        An unknown name is reported on stderr and otherwise ignored.
        """
        if isinstance(app, str):
            for i in range(int(AppType.APP_COUNT)):
                if app_type_to_string(AppType(i)) == app:
                    self.block_app(AppType(i))
                    return
            print(f"[DPIEngine] Unknown app: {app}", file=sys.stderr)
            return

        if self._rule_manager is not None:
            self._rule_manager.block_app(app)

    def unblock_app(self, app: AppType | str) -> None:
        """Unblock an application.  Mirrors both ``unblockApp`` overloads.

        Unlike :meth:`block_app`, an unknown name is silently ignored here —
        the C++ string overload has no ``else`` branch.  Preserved.
        """
        if isinstance(app, str):
            for i in range(int(AppType.APP_COUNT)):
                if app_type_to_string(AppType(i)) == app:
                    self.unblock_app(AppType(i))
                    return
            return

        if self._rule_manager is not None:
            self._rule_manager.unblock_app(app)

    def block_domain(self, domain: str) -> None:
        """Block a domain or wildcard pattern.  Mirrors ``blockDomain``."""
        if self._rule_manager is not None:
            self._rule_manager.block_domain(domain)

    def unblock_domain(self, domain: str) -> None:
        """Unblock a domain or pattern.  Mirrors ``unblockDomain``."""
        if self._rule_manager is not None:
            self._rule_manager.unblock_domain(domain)

    def load_rules(self, filename: str | Path) -> bool:
        """Load rules from a file.  Mirrors ``loadRules``."""
        if self._rule_manager is not None:
            return self._rule_manager.load_rules(filename)
        return False

    def save_rules(self, filename: str | Path) -> bool:
        """Save rules to a file.  Mirrors ``saveRules``."""
        if self._rule_manager is not None:
            return self._rule_manager.save_rules(filename)
        return False

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def generate_report(self) -> str:
        """Render the engine statistics report.

        Mirrors ``generateReport`` character for character.  Sections for the
        LB manager, FP manager and rule manager are emitted only when those
        components exist, as in the original.
        """
        s = self._stats
        total_packets = s.total_packets.get()
        dropped = s.dropped_packets.get()

        lines: list[str] = []
        lines.append("\n╔══════════════════════════════════════════════════════════════╗")
        lines.append("║                    DPI ENGINE STATISTICS                      ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        lines.append("║ PACKET STATISTICS                                             ║")
        lines.append(f"║   Total Packets:      {total_packets:>12}                        ║")
        lines.append(f"║   Total Bytes:        {s.total_bytes.get():>12}                        ║")
        lines.append(f"║   TCP Packets:        {s.tcp_packets.get():>12}                        ║")
        lines.append(f"║   UDP Packets:        {s.udp_packets.get():>12}                        ║")

        lines.append("╠══════════════════════════════════════════════════════════════╣")
        lines.append("║ FILTERING STATISTICS                                          ║")
        lines.append(f"║   Forwarded:          {s.forwarded_packets.get():>12}                        ║")
        lines.append(f"║   Dropped/Blocked:    {dropped:>12}                        ║")

        if total_packets > 0:
            drop_rate = 100.0 * dropped / total_packets
            lines.append(f"║   Drop Rate:          {drop_rate:>11.2f}%                        ║")

        if self._lb_manager is not None:
            lb_stats = self._lb_manager.get_aggregated_stats()
            lines.append("╠══════════════════════════════════════════════════════════════╣")
            lines.append("║ LOAD BALANCER STATISTICS                                      ║")
            lines.append(f"║   LB Received:        {lb_stats.total_received:>12}                        ║")
            lines.append(f"║   LB Dispatched:      {lb_stats.total_dispatched:>12}                        ║")

        if self._fp_manager is not None:
            fp_stats = self._fp_manager.get_aggregated_stats()
            lines.append("╠══════════════════════════════════════════════════════════════╣")
            lines.append("║ FAST PATH STATISTICS                                          ║")
            lines.append(f"║   FP Processed:       {fp_stats.total_processed:>12}                        ║")
            lines.append(f"║   FP Forwarded:       {fp_stats.total_forwarded:>12}                        ║")
            lines.append(f"║   FP Dropped:         {fp_stats.total_dropped:>12}                        ║")
            lines.append(f"║   Active Connections: {fp_stats.total_connections:>12}                        ║")

        if self._rule_manager is not None:
            rule_stats = self._rule_manager.get_stats()
            lines.append("╠══════════════════════════════════════════════════════════════╣")
            lines.append("║ BLOCKING RULES                                                ║")
            lines.append(f"║   Blocked IPs:        {rule_stats.blocked_ips:>12}                        ║")
            lines.append(f"║   Blocked Apps:       {rule_stats.blocked_apps:>12}                        ║")
            lines.append(f"║   Blocked Domains:    {rule_stats.blocked_domains:>12}                        ║")
            lines.append(f"║   Blocked Ports:      {rule_stats.blocked_ports:>12}                        ║")

        lines.append("╚══════════════════════════════════════════════════════════════╝")

        return "\n".join(lines) + "\n"

    def generate_classification_report(self) -> str:
        """Delegate to the FP manager.  Mirrors ``generateClassificationReport``."""
        if self._fp_manager is not None:
            return self._fp_manager.generate_classification_report()
        return ""

    def get_stats(self) -> DPIStats:
        """Return the live statistics block.  Mirrors ``getStats``."""
        return self._stats

    def print_status(self) -> None:
        """Print a one-line live status.  Mirrors ``printStatus``."""
        print("\n--- Live Status ---")
        print(
            f"Packets: {self._stats.total_packets.get()}"
            f" | Forwarded: {self._stats.forwarded_packets.get()}"
            f" | Dropped: {self._stats.dropped_packets.get()}"
        )

        if self._fp_manager is not None:
            fp_stats = self._fp_manager.get_aggregated_stats()
            print(f"Connections: {fp_stats.total_connections}")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_rule_manager(self) -> RuleManager:
        """Return the rule manager.  Mirrors ``getRuleManager``."""
        assert self._rule_manager is not None, "call initialize() first"
        return self._rule_manager

    def get_config(self) -> Config:
        """Return the configuration.  Mirrors ``getConfig``."""
        return self._config

    def get_flow_snapshot(self) -> FlowSnapshot:
        """Return a read-only view of the results of the last run.

        No C++ counterpart.  Added so external consumers can read flow records
        and statistics without touching private attributes.  **Read-only**: it
        gathers already-computed values and mutates nothing.

        Call after :meth:`process_file` has returned.  The connection trackers
        are not cleared by :meth:`stop`, so flow records remain available.
        Returns an empty snapshot if :meth:`initialize` was never called.

        The returned ``Connection`` objects are the live records, not copies --
        treat them as read-only.
        """
        connections: list[Connection] = []
        if self._fp_manager is not None:
            for i in range(self._fp_manager.get_num_fps()):
                connections.extend(
                    self._fp_manager.get_fp(i).get_connection_tracker().get_all_connections()
                )

        app_distribution: dict[AppType, int] = {}
        top_domains: tuple[tuple[str, int], ...] = ()
        if self._global_conn_table is not None:
            global_stats = self._global_conn_table.get_global_stats()
            app_distribution = dict(global_stats.app_distribution)
            top_domains = tuple(global_stats.top_domains)

        return FlowSnapshot(
            connections=tuple(connections),
            packet_stats=self._stats.snapshot(),
            app_distribution=app_distribution,
            top_domains=top_domains,
            rule_stats=(
                self._rule_manager.get_stats() if self._rule_manager is not None else None
            ),
            num_load_balancers=self._config.num_load_balancers,
            fps_per_lb=self._config.fps_per_lb,
        )

    def is_running(self) -> bool:
        """Return whether the engine is running.  Mirrors ``isRunning``."""
        return self._running.is_set()

    # ------------------------------------------------------------------
    # Context manager (stands in for the C++ destructor)
    # ------------------------------------------------------------------
    def __enter__(self) -> "DPIEngine":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"DPIEngine(lbs={self._config.num_load_balancers}, "
            f"fps_per_lb={self._config.fps_per_lb}, running={self.is_running()})"
        )

    # C++-style aliases
    processFile = process_file
    waitForCompletion = wait_for_completion
    createPacketJob = create_packet_job
    blockIP = block_ip
    unblockIP = unblock_ip
    blockApp = block_app
    unblockApp = unblock_app
    blockDomain = block_domain
    unblockDomain = unblock_domain
    loadRules = load_rules
    saveRules = save_rules
    generateReport = generate_report
    generateClassificationReport = generate_classification_report
    getStats = get_stats
    printStatus = print_status
    getRuleManager = get_rule_manager
    getConfig = get_config
    getFlowSnapshot = get_flow_snapshot
    isRunning = is_running
