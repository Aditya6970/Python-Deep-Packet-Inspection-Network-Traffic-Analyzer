#!/usr/bin/env python3
"""Multi-threaded DPI Engine — Fixed Version.

Python port of ``src/dpi_mt.cpp``.
Architecture: Reader -> LB threads -> FP threads -> Output.

This file is a **self-contained duplicate** of the threaded engine.  In C++ it
re-implements its own ``TSQueue``, ``Rules``, ``FastPath``, ``LoadBalancer``
and ``DPIEngine`` rather than including ``dpi_engine.h``, and this port keeps
that separation so the two engines can drift independently exactly as the
originals do.

How it differs from ``main_dpi`` (all preserved):

* ``Rules`` uses **substring** domain matching, no wildcards, no port rules.
* ``FastPath`` latches on ``classified``; SNI is attempted only on port 443
  with ``payload_length > 5``, HTTP Host only on port 80 with ``> 10``.
* Its ``TSQueue.pop`` always takes a timeout (no infinite-blocking overload).
* It has no connection state machine and no stale-connection cleanup.
* The output writer polls at 50 ms rather than 100 ms.

C++ concepts replaced
---------------------
``TSQueue<T>``
    Reuses :class:`~dpi.thread_safe_queue.ThreadSafeQueue`, whose semantics are
    identical for the operations this file uses.  The C++ duplicate differs
    only in that its ``pop`` requires a timeout.

``std::thread`` / ``std::atomic``
    Become :class:`threading.Thread` / :class:`threading.Event` and
    :class:`~dpi.types.AtomicCounter`.
"""

from __future__ import annotations

import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from dpi.packet_parser import PacketParser
from dpi.pcap_reader import PcapReader
from dpi.sni_extractor import HTTPHostExtractor, SNIExtractor
from dpi.thread_safe_queue import ThreadSafeQueue
from dpi.types import AppType, AtomicCounter, FiveTuple, app_type_to_string, five_tuple_hash, sni_to_app_type

_PACKET_HEADER_FMT = "=IIII"
_GLOBAL_HEADER_FMT = "=IHHiIII"


# =============================================================================
# Packet Job - Contains all packet data (self-contained, no pointers)
# =============================================================================
@dataclass(slots=True)
class Packet:
    """Mirrors ``struct Packet``."""

    id: int = 0
    ts_sec: int = 0
    ts_usec: int = 0
    tuple: FiveTuple | None = None
    data: bytes = b""
    tcp_flags: int = 0
    payload_offset: int = 0
    payload_length: int = 0


# =============================================================================
# Flow Entry
# =============================================================================
@dataclass(slots=True)
class FlowEntry:
    """Mirrors ``struct FlowEntry``."""

    tuple: FiveTuple | None = None
    app_type: AppType = AppType.UNKNOWN
    sni: str = ""
    packets: int = 0
    bytes: int = 0
    blocked: bool = False
    classified: bool = False


def _parse_ip(ip: str) -> int:
    """Mirrors the ``parseIP`` lambda (the fifth copy of this routine)."""
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


# =============================================================================
# Blocking Rules
# =============================================================================
class Rules:
    """Mirrors ``class Rules`` — substring domain matching, no wildcards."""

    __slots__ = ("_lock", "_blocked_ips", "_blocked_apps", "_blocked_domains")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._blocked_ips: set[int] = set()
        self._blocked_apps: set[AppType] = set()
        self._blocked_domains: list[str] = []

    def block_ip(self, ip: str) -> None:
        with self._lock:
            self._blocked_ips.add(_parse_ip(ip))
        print(f"[Rules] Blocked IP: {ip}")

    def block_app(self, app: str) -> None:
        with self._lock:
            for i in range(int(AppType.APP_COUNT)):
                if app_type_to_string(AppType(i)) == app:
                    self._blocked_apps.add(AppType(i))
                    print(f"[Rules] Blocked app: {app}")
                    return
        print(f"[Rules] Unknown app: {app}", file=sys.stderr)

    def block_domain(self, domain: str) -> None:
        with self._lock:
            self._blocked_domains.append(domain)
        print(f"[Rules] Blocked domain: {domain}")

    def is_blocked(self, src_ip: int, app: AppType, sni: str) -> bool:
        with self._lock:
            if src_ip in self._blocked_ips:
                return True
            if app in self._blocked_apps:
                return True
            return any(dom in sni for dom in self._blocked_domains)


# =============================================================================
# Statistics (thread-safe)
# =============================================================================
class Stats:
    """Mirrors ``struct Stats``."""

    __slots__ = (
        "total_packets", "total_bytes", "forwarded", "dropped",
        "tcp_packets", "udp_packets", "app_lock", "app_counts", "detected_snis",
    )

    def __init__(self) -> None:
        self.total_packets = AtomicCounter()
        self.total_bytes = AtomicCounter()
        self.forwarded = AtomicCounter()
        self.dropped = AtomicCounter()
        self.tcp_packets = AtomicCounter()
        self.udp_packets = AtomicCounter()

        # Per-app stats (protected by a mutex)
        self.app_lock = threading.Lock()
        self.app_counts: dict[AppType, int] = {}
        self.detected_snis: dict[str, AppType] = {}

    def record_app(self, app: AppType, sni: str) -> None:
        with self.app_lock:
            self.app_counts[app] = self.app_counts.get(app, 0) + 1
            if sni:
                self.detected_snis[sni] = app


# =============================================================================
# Fast Path Processor (one per FP thread)
# =============================================================================
class FastPath:
    """Mirrors ``class FastPath``."""

    __slots__ = ("_id", "_rules", "_stats", "_output_queue", "_input_queue",
                 "_flows", "_running", "_thread", "_processed")

    def __init__(self, fp_id: int, rules: Rules, stats: Stats,
                 output_queue: ThreadSafeQueue[Packet]) -> None:
        self._id = fp_id
        self._rules = rules
        self._stats = stats
        self._output_queue = output_queue
        self._input_queue: ThreadSafeQueue[Packet] = ThreadSafeQueue()
        self._flows: dict[FiveTuple, FlowEntry] = {}
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._processed = AtomicCounter()

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, name=f"mtFP{self._id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._input_queue.shutdown()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        self._thread = None

    def queue(self) -> ThreadSafeQueue[Packet]:
        return self._input_queue

    def processed(self) -> int:
        return self._processed.get()

    def _run(self) -> None:
        while self._running.is_set():
            pkt = self._input_queue.pop_with_timeout(100)
            if pkt is None:
                continue

            self._processed.increment()

            # Get or create flow (C++ operator[] default-constructs)
            flow = self._flows.get(pkt.tuple)
            if flow is None:
                flow = FlowEntry()
                self._flows[pkt.tuple] = flow
            if flow.packets == 0:
                flow.tuple = pkt.tuple
            flow.packets += 1
            flow.bytes += len(pkt.data)

            # Try to classify if not done yet
            if not flow.classified:
                self._classify_flow(pkt, flow)

            # Check blocking
            if not flow.blocked:
                flow.blocked = self._rules.is_blocked(
                    pkt.tuple.src_ip, flow.app_type, flow.sni
                )

            # Record stats
            self._stats.record_app(flow.app_type, flow.sni)

            # Forward or drop
            if flow.blocked:
                self._stats.dropped.increment()
            else:
                self._stats.forwarded.increment()
                self._output_queue.push(pkt)

    def _classify_flow(self, pkt: Packet, flow: FlowEntry) -> None:
        """Mirrors ``classifyFlow``."""
        # Try SNI extraction for HTTPS
        if pkt.tuple.dst_port == 443 and pkt.payload_length > 5:
            payload = memoryview(pkt.data)[pkt.payload_offset :]
            sni = SNIExtractor.extract(payload, pkt.payload_length)
            if sni is not None:
                flow.sni = sni
                flow.app_type = sni_to_app_type(sni)
                flow.classified = True
                return

        # Try HTTP Host extraction
        if pkt.tuple.dst_port == 80 and pkt.payload_length > 10:
            payload = memoryview(pkt.data)[pkt.payload_offset :]
            host = HTTPHostExtractor.extract(payload, pkt.payload_length)
            if host is not None:
                flow.sni = host
                flow.app_type = sni_to_app_type(host)
                flow.classified = True
                return

        # DNS
        if pkt.tuple.dst_port == 53 or pkt.tuple.src_port == 53:
            flow.app_type = AppType.DNS
            flow.classified = True
            return

        # Port-based fallback (but don't mark as classified - might get SNI later)
        if pkt.tuple.dst_port == 443:
            flow.app_type = AppType.HTTPS
        elif pkt.tuple.dst_port == 80:
            flow.app_type = AppType.HTTP


# =============================================================================
# Load Balancer (one per LB thread)
# =============================================================================
class LoadBalancer:
    """Mirrors ``class LoadBalancer`` (the dpi_mt one, not load_balancer.h)."""

    __slots__ = ("_id", "_fps", "_num_fps", "_input_queue", "_running",
                 "_thread", "_dispatched")

    def __init__(self, lb_id: int, fps: list[FastPath]) -> None:
        self._id = lb_id
        self._fps = list(fps)
        self._num_fps = len(fps)
        self._input_queue: ThreadSafeQueue[Packet] = ThreadSafeQueue()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._dispatched = AtomicCounter()

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, name=f"mtLB{self._id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._input_queue.shutdown()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        self._thread = None

    def queue(self) -> ThreadSafeQueue[Packet]:
        return self._input_queue

    def dispatched(self) -> int:
        return self._dispatched.get()

    def _run(self) -> None:
        while self._running.is_set():
            pkt = self._input_queue.pop_with_timeout(100)
            if pkt is None:
                continue

            # Hash to select FP -- same shared-hash skew as load_balancer.py
            fp_idx = five_tuple_hash(pkt.tuple) % self._num_fps

            self._fps[fp_idx].queue().push(pkt)
            self._dispatched.increment()


# =============================================================================
# DPI Engine
# =============================================================================
@dataclass(slots=True)
class Config:
    """Mirrors ``struct DPIEngine::Config``."""

    num_lbs: int = 2
    fps_per_lb: int = 2


class DPIEngine:
    """Mirrors ``class DPIEngine`` (the dpi_mt one)."""

    __slots__ = ("_config", "_rules", "_stats", "_output_queue", "_fps", "_lbs")

    def __init__(self, cfg: Config) -> None:
        self._config = cfg
        self._rules = Rules()
        self._stats = Stats()
        self._output_queue: ThreadSafeQueue[Packet] = ThreadSafeQueue()

        total_fps = cfg.num_lbs * cfg.fps_per_lb

        print("")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║              DPI ENGINE v2.0 (Multi-threaded)                 ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print(
            f"║ Load Balancers: {cfg.num_lbs:>2}"
            f"    FPs per LB: {cfg.fps_per_lb:>2}"
            f"    Total FPs: {total_fps:>2}     ║"
        )
        print("╚══════════════════════════════════════════════════════════════╝\n")

        # Create FP threads
        self._fps = [
            FastPath(i, self._rules, self._stats, self._output_queue)
            for i in range(total_fps)
        ]

        # Create LB threads, each managing a subset of FPs
        self._lbs = [
            LoadBalancer(lb, self._fps[lb * cfg.fps_per_lb : (lb + 1) * cfg.fps_per_lb])
            for lb in range(cfg.num_lbs)
        ]

    def block_ip(self, ip: str) -> None:
        self._rules.block_ip(ip)

    def block_app(self, app: str) -> None:
        self._rules.block_app(app)

    def block_domain(self, dom: str) -> None:
        self._rules.block_domain(dom)

    def process(self, input_file: str, output_file: str) -> bool:
        """Mirrors ``process``."""
        # Open input
        reader = PcapReader()
        if not reader.open(input_file):
            return False

        # Open output
        try:
            output = Path(output_file).open("wb")
        except OSError:
            print("Cannot open output file", file=sys.stderr)
            return False

        # Write PCAP header
        h = reader.get_global_header()
        output.write(
            struct.pack(
                _GLOBAL_HEADER_FMT,
                h.magic_number, h.version_major, h.version_minor,
                h.thiszone, h.sigfigs, h.snaplen, h.network,
            )
        )

        # Start all threads
        for fp in self._fps:
            fp.start()
        for lb in self._lbs:
            lb.start()

        # Start output writer thread
        output_running = threading.Event()
        output_running.set()
        output_lock = threading.Lock()

        def output_worker() -> None:
            while output_running.is_set() or self._output_queue.size() > 0:
                pkt = self._output_queue.pop_with_timeout(50)
                if pkt is None:
                    continue
                size = len(pkt.data)
                with output_lock:
                    output.write(
                        struct.pack(_PACKET_HEADER_FMT, pkt.ts_sec, pkt.ts_usec, size, size)
                    )
                    output.write(pkt.data)

        output_thread = threading.Thread(target=output_worker, name="mtOutput", daemon=True)
        output_thread.start()

        # Read and dispatch packets
        print("[Reader] Processing packets...")
        pkt_id = 0

        while True:
            raw = reader.read_next_packet()
            if raw is None:
                break

            parsed = PacketParser.parse(raw)
            if parsed is None:
                continue
            if not parsed.has_ip or (not parsed.has_tcp and not parsed.has_udp):
                continue

            # Create packet
            pkt = Packet()
            pkt.id = pkt_id
            pkt_id += 1
            pkt.ts_sec = raw.header.ts_sec
            pkt.ts_usec = raw.header.ts_usec
            pkt.tcp_flags = parsed.tcp_flags
            pkt.data = raw.data

            # Parse 5-tuple
            pkt.tuple = FiveTuple(
                src_ip=_parse_ip(parsed.src_ip),
                dst_ip=_parse_ip(parsed.dest_ip),
                src_port=parsed.src_port,
                dst_port=parsed.dest_port,
                protocol=parsed.protocol,
            )

            # Calculate payload offset
            pkt.payload_offset = 14  # Ethernet
            if len(pkt.data) > 14:
                ip_ihl = pkt.data[14] & 0x0F
                pkt.payload_offset += ip_ihl * 4

                if parsed.has_tcp and pkt.payload_offset + 12 < len(pkt.data):
                    tcp_off = (pkt.data[pkt.payload_offset + 12] >> 4) & 0x0F
                    pkt.payload_offset += tcp_off * 4
                elif parsed.has_udp:
                    pkt.payload_offset += 8

                if pkt.payload_offset < len(pkt.data):
                    pkt.payload_length = len(pkt.data) - pkt.payload_offset
                else:
                    pkt.payload_length = 0

            # Update stats
            self._stats.total_packets.increment()
            self._stats.total_bytes.add(len(pkt.data))
            if parsed.has_tcp:
                self._stats.tcp_packets.increment()
            elif parsed.has_udp:
                self._stats.udp_packets.increment()

            # Dispatch to LB (hash-based)
            lb_idx = five_tuple_hash(pkt.tuple) % len(self._lbs)
            self._lbs[lb_idx].queue().push(pkt)

        print(f"[Reader] Done reading {pkt_id} packets")
        reader.close()

        # Wait for queues to drain (fixed sleep, as in the original)
        time.sleep(0.5)

        # Stop all threads
        for lb in self._lbs:
            lb.stop()
        for fp in self._fps:
            fp.stop()

        output_running.clear()
        self._output_queue.shutdown()
        output_thread.join()

        output.close()

        # Print report
        self._print_report()

        return True

    def _print_report(self) -> None:
        """Mirrors ``printReport``."""
        s = self._stats
        print("")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║                      PROCESSING REPORT                        ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print(f"║ Total Packets:      {s.total_packets.get():>12}                           ║")
        print(f"║ Total Bytes:        {s.total_bytes.get():>12}                           ║")
        print(f"║ TCP Packets:        {s.tcp_packets.get():>12}                           ║")
        print(f"║ UDP Packets:        {s.udp_packets.get():>12}                           ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print(f"║ Forwarded:          {s.forwarded.get():>12}                           ║")
        print(f"║ Dropped:            {s.dropped.get():>12}                           ║")

        # Thread stats
        print("╠══════════════════════════════════════════════════════════════╣")
        print("║ THREAD STATISTICS                                             ║")
        for i, lb in enumerate(self._lbs):
            print(f"║   LB{i} dispatched:   {lb.dispatched():>12}                           ║")
        for i, fp in enumerate(self._fps):
            print(f"║   FP{i} processed:    {fp.processed():>12}                           ║")

        # App distribution
        print("╠══════════════════════════════════════════════════════════════╣")
        print("║                   APPLICATION BREAKDOWN                       ║")
        print("╠══════════════════════════════════════════════════════════════╣")

        with s.app_lock:
            sorted_apps = sorted(s.app_counts.items(), key=lambda kv: -kv[1])
            total = s.total_packets.get()

            for app, count in sorted_apps:
                pct = (100.0 * count / total) if total > 0 else 0.0
                bar_str = "#" * int(pct / 5)
                print(
                    f"║ {app_type_to_string(app):<15}{count:>8}"
                    f" {pct:>5.1f}% {bar_str:<20}  ║"
                )

            print("╚══════════════════════════════════════════════════════════════╝")

            # Detected SNIs
            if s.detected_snis:
                print("\n[Detected Domains/SNIs]")
                for sni, app in s.detected_snis.items():
                    print(f"  - {sni} -> {app_type_to_string(app)}")


# =============================================================================
# Main
# =============================================================================
def print_usage(prog: str) -> None:
    """Mirrors ``printUsage``."""
    print(f"""
DPI Engine v2.0 - Multi-threaded Deep Packet Inspection
========================================================

Usage: {prog} <input.pcap> <output.pcap> [options]

Options:
  --block-ip <ip>        Block source IP
  --block-app <app>      Block application (YouTube, Facebook, etc.)
  --block-domain <dom>   Block domain (substring match)
  --lbs <n>              Number of load balancer threads (default: 2)
  --fps <n>              FP threads per LB (default: 2)

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

    cfg = Config()
    block_ips: list[str] = []
    block_apps: list[str] = []
    block_domains: list[str] = []

    i = 3
    while i < argc:
        arg = argv[i]
        if arg == "--block-ip" and i + 1 < argc:
            i += 1
            block_ips.append(argv[i])
        elif arg == "--block-app" and i + 1 < argc:
            i += 1
            block_apps.append(argv[i])
        elif arg == "--block-domain" and i + 1 < argc:
            i += 1
            block_domains.append(argv[i])
        elif arg == "--lbs" and i + 1 < argc:
            i += 1
            cfg.num_lbs = int(argv[i])
        elif arg == "--fps" and i + 1 < argc:
            i += 1
            cfg.fps_per_lb = int(argv[i])
        i += 1

    engine = DPIEngine(cfg)

    for ip in block_ips:
        engine.block_ip(ip)
    for app in block_apps:
        engine.block_app(app)
    for dom in block_domains:
        engine.block_domain(dom)

    if not engine.process(input_file, output_file):
        return 1

    print(f"\nOutput written to: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
