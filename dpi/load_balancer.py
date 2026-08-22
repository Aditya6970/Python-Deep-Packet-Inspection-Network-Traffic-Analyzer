"""Load-balancer threads: hash flows onto fast-path workers.

Python port of ``include/load_balancer.h`` + ``src/load_balancer.cpp``
(C++ ``namespace DPI``).

Architecture::

    Reader Thread -> LB Queues -> LB Threads -> FP Queues -> FP Threads

Each LB thread pops packets from its input queue, hashes the five-tuple, and
forwards to one of the FP queues it serves.  Consistent hashing is what keeps a
flow pinned to a single FP, which is what lets
:class:`~dpi.connection_tracker.ConnectionTracker` run without locks.

C++ concepts replaced
---------------------
``std::thread`` + ``std::atomic<bool> running_``
    Become :class:`threading.Thread` and :class:`threading.Event`.  ``Event``
    gives the atomic load/store the ``std::atomic<bool>`` provided, without
    relying on attribute access being atomic under free-threading.

``std::vector<ThreadSafeQueue<PacketJob>*>`` (non-owning pointers)
    Becomes a plain ``list`` of queue objects.  Python references are already
    non-owning in the sense that matters: the FP owns its queue, the LB just
    holds a reference.

``std::unique_ptr<LoadBalancer>`` in ``LBManager``
    Becomes ordinary object references; ``stop_all()`` in an explicit
    ``__del__``-free flow replaces the destructor chain.

Fixed upstream bug
------------------
The C++ applied ``hash % n`` to the *same* five-tuple hash at both balancing
levels, so whenever ``num_lbs == fps_per_lb`` — including the **default 2x2** —
only the diagonal FPs received any traffic and half the configured threads sat
idle.  :meth:`LoadBalancer.select_fp` now avalanches the hash before the second
modulo, which spreads flows across every FP at any configuration.  Flow
affinity is unchanged: a given five-tuple still maps to exactly one FP, which
is what keeps :class:`~dpi.connection_tracker.ConnectionTracker` lock-free.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Final

from .thread_safe_queue import ThreadSafeQueue
from .types import FiveTuple, PacketJob, flow_hash

__all__ = ["LB_QUEUE_SIZE", "LB_POP_TIMEOUT_MS", "mix64", "LBStats", "AggregatedLBStats", "LoadBalancer", "LBManager"]

#: Input queue capacity, hard-coded in the C++ constructor's initialiser list.
LB_QUEUE_SIZE: Final[int] = 10000
#: Poll interval so the run loop can observe the running flag.
LB_POP_TIMEOUT_MS: Final[float] = 100.0

_UINT64_MASK: Final[int] = 0xFFFFFFFFFFFFFFFF


def mix64(value: int) -> int:
    """Avalanche a 64-bit value (the SplitMix64 finalizer).

    Used to decorrelate the second level of load balancing from the first —
    see :meth:`LoadBalancer.select_fp`.  Every input bit affects every output
    bit, so ``mix64(h) % n`` is independent of ``h % m`` for the small ``n``
    and ``m`` this engine uses.
    """
    value &= _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


@dataclass(slots=True)
class LBStats:
    """Per-LB counters.  Mirrors ``LoadBalancer::LBStats``."""

    packets_received: int = 0
    packets_dispatched: int = 0
    per_fp_packets: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AggregatedLBStats:
    """Totals across LBs.  Mirrors ``LBManager::AggregatedStats``."""

    total_received: int = 0
    total_dispatched: int = 0


# ============================================================================
# Load Balancer Thread
# ============================================================================
class LoadBalancer:
    """One load-balancer thread serving a fixed pool of FP queues.

    Mirrors ``class LoadBalancer``.
    """

    __slots__ = (
        "_lb_id",
        "_fp_start_id",
        "_num_fps",
        "_input_queue",
        "_fp_queues",
        "_packets_received",
        "_packets_dispatched",
        "_per_fp_counts",
        "_running",
        "_thread",
    )

    def __init__(
        self,
        lb_id: int,
        fp_queues: list[ThreadSafeQueue[PacketJob]],
        fp_start_id: int,
    ) -> None:
        self._lb_id = lb_id
        self._fp_start_id = fp_start_id
        self._num_fps = len(fp_queues)

        # Input queue from reader
        self._input_queue: ThreadSafeQueue[PacketJob] = ThreadSafeQueue(LB_QUEUE_SIZE)

        # Output queues to FP threads
        self._fp_queues = list(fp_queues)

        # Statistics.  per_fp_counts is touched only by this LB's own thread,
        # so it needs no synchronisation -- as the C++ comment notes.
        self._packets_received = 0
        self._packets_dispatched = 0
        self._per_fp_counts = [0] * len(fp_queues)

        # Thread control
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the LB thread.  Mirrors ``start()``; a no-op if already running."""
        if self._running.is_set():
            return

        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name=f"LB{self._lb_id}", daemon=True
        )
        self._thread.start()

        print(
            f"[LB{self._lb_id}] Started (serving FP{self._fp_start_id}"
            f"-FP{self._fp_start_id + self._num_fps - 1})"
        )

    def stop(self) -> None:
        """Stop the LB thread and join it.  Mirrors ``stop()``.

        NOTE: the run loop can be parked inside ``fp_queue.push()`` when a
        downstream FP queue is full.  Shutting down the *input* queue does not
        wake that wait, so the join can block until the FP drains.  The C++ has
        the same exposure; :class:`~dpi.dpi_engine.DPIEngine` therefore stops
        LBs before FPs.
        """
        if not self._running.is_set():
            return

        self._running.clear()
        self._input_queue.shutdown()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        self._thread = None

        print(f"[LB{self._lb_id}] Stopped")

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------
    def _run(self) -> None:
        """Pop, hash, dispatch.  Mirrors ``LoadBalancer::run``."""
        while self._running.is_set():
            # Get packet from input queue (with timeout to check running flag)
            job = self._input_queue.pop_with_timeout(LB_POP_TIMEOUT_MS)

            if job is None:
                continue  # Timeout or shutdown

            self._packets_received += 1

            # Select target FP based on five-tuple hash
            fp_index = self.select_fp(job.tuple)

            # Push to selected FP's queue
            self._fp_queues[fp_index].push(job)

            self._packets_dispatched += 1
            self._per_fp_counts[fp_index] += 1

    def select_fp(self, tuple_: FiveTuple) -> int:
        """Choose an FP index within this LB's pool.  Mirrors ``selectFP``.

        FIXED (was UPSTREAM BUG — uneven distribution).  The C++ was::

            return hasher(tuple) % num_fps_;

        while :meth:`LBManager.get_lb_for_packet` picked the LB with
        ``hasher(tuple) % num_lbs``.  Two moduli over the *same* hash are
        perfectly correlated whenever the divisors share a factor: with the
        default ``--lbs 2 --fps 2`` both expressions produce the identical
        value, so LB0 only ever fed its FP0 and LB1 only ever fed its FP1.
        LB0's FP1 and LB1's FP0 received nothing — half the engine idle.
        (Generally only FPs with ``fp_index == lb_index (mod gcd)`` were
        reachable.)

        Passing the hash through :func:`mix64` first breaks that correlation,
        because the finalizer makes every output bit depend on every input bit.
        The choice stays a pure function of the five-tuple, so flow affinity —
        the property the lock-free connection tracker depends on — is intact.

        It also uses :func:`~dpi.types.flow_hash` rather than the raw
        ``five_tuple_hash``, so a conversation's two directions land on the
        *same* FP and can share one ``Connection``.
        """
        return mix64(flow_hash(tuple_)) % self._num_fps


    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_input_queue(self) -> ThreadSafeQueue[PacketJob]:
        """Return the input queue for the reader to push into."""
        return self._input_queue

    def get_stats(self) -> LBStats:
        """Return this LB's counters.  Mirrors ``getStats``."""
        return LBStats(
            packets_received=self._packets_received,
            packets_dispatched=self._packets_dispatched,
            per_fp_packets=list(self._per_fp_counts),
        )

    def get_id(self) -> int:
        """Return this LB's id.  Mirrors ``getId``."""
        return self._lb_id

    def is_running(self) -> bool:
        """Return whether the thread is running.  Mirrors ``isRunning``."""
        return self._running.is_set()

    def __repr__(self) -> str:
        return (
            f"LoadBalancer(id={self._lb_id}, fps={self._num_fps}, "
            f"running={self.is_running()}, dispatched={self._packets_dispatched})"
        )

    # C++-style aliases
    selectFP = select_fp
    getInputQueue = get_input_queue
    getStats = get_stats
    getId = get_id
    isRunning = is_running


# ============================================================================
# LB Manager - creates and manages multiple LB threads
# ============================================================================
class LBManager:
    """Creates and supervises a pool of :class:`LoadBalancer` threads.

    Mirrors ``class LBManager``.
    """

    __slots__ = ("_lbs", "_fps_per_lb")

    def __init__(
        self,
        num_lbs: int,
        fps_per_lb: int,
        fp_queues: list[ThreadSafeQueue[PacketJob]],
    ) -> None:
        """Partition ``fp_queues`` into contiguous per-LB slices.

        LB *i* serves queues ``[i * fps_per_lb, (i + 1) * fps_per_lb)``.

        Requires ``len(fp_queues) >= num_lbs * fps_per_lb``; the C++ indexes
        ``fp_queues[fp_start + i]`` with no bounds check and reads out of
        bounds if that does not hold.  Python raises :class:`IndexError`
        instead, which is the same defect surfaced safely.
        """
        self._fps_per_lb = fps_per_lb
        self._lbs: list[LoadBalancer] = []

        # Create load balancers, each handling a subset of FPs
        for lb_id in range(num_lbs):
            fp_start = lb_id * fps_per_lb
            lb_fp_queues = [fp_queues[fp_start + i] for i in range(fps_per_lb)]
            self._lbs.append(LoadBalancer(lb_id, lb_fp_queues, fp_start))

        print(f"[LBManager] Created {num_lbs} load balancers, {fps_per_lb} FPs each")

    def start_all(self) -> None:
        """Start every LB thread.  Mirrors ``startAll``."""
        for lb in self._lbs:
            lb.start()

    def stop_all(self) -> None:
        """Stop every LB thread.  Mirrors ``stopAll``."""
        for lb in self._lbs:
            lb.stop()

    def get_lb_for_packet(self, tuple_: FiveTuple) -> LoadBalancer:
        """First-level balancing: pick an LB from the five-tuple hash.

        Mirrors ``getLBForPacket``, but on :func:`~dpi.types.flow_hash` so both
        directions of a conversation reach the same LB.  See
        :meth:`LoadBalancer.select_fp` for how the second level is decorrelated
        from this one.
        """
        return self._lbs[flow_hash(tuple_) % len(self._lbs)]

    def get_lb(self, lb_id: int) -> LoadBalancer:
        """Return a specific LB.  Mirrors ``getLB``."""
        return self._lbs[lb_id]

    def get_num_lbs(self) -> int:
        """Return the number of LB threads.  Mirrors ``getNumLBs``."""
        return len(self._lbs)

    def get_aggregated_stats(self) -> AggregatedLBStats:
        """Sum counters across LBs.  Mirrors ``getAggregatedStats``."""
        total_received = 0
        total_dispatched = 0
        for lb in self._lbs:
            s = lb.get_stats()
            total_received += s.packets_received
            total_dispatched += s.packets_dispatched
        return AggregatedLBStats(
            total_received=total_received, total_dispatched=total_dispatched
        )

    def __len__(self) -> int:
        return len(self._lbs)

    def __repr__(self) -> str:
        s = self.get_aggregated_stats()
        return (
            f"LBManager(lbs={len(self._lbs)}, fps_per_lb={self._fps_per_lb}, "
            f"dispatched={s.total_dispatched})"
        )

    # C++-style aliases
    startAll = start_all
    stopAll = stop_all
    getLBForPacket = get_lb_for_packet
    getLB = get_lb
    getNumLBs = get_num_lbs
    getAggregatedStats = get_aggregated_stats


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import io
    import time
    from contextlib import redirect_stdout

    def tup(i: int) -> FiveTuple:
        return FiveTuple(0x0A01A8C0, 0x08080808, 1024 + i, 443, 6)

    # ---- every FP is now reachable at every configuration --------------
    print("FP reachability by configuration (flows spread over 4000 tuples):")
    for num_lbs, fps_per_lb in ((2, 2), (2, 3), (3, 3), (4, 2), (1, 4), (2, 4), (4, 4)):
        hits: dict[tuple[int, int], int] = {}
        for i in range(4000):
            h = flow_hash(tup(i))
            key = (h % num_lbs, mix64(h) % fps_per_lb)
            hits[key] = hits.get(key, 0) + 1
        total_fps = num_lbs * fps_per_lb
        used = len(hits)
        spread = min(hits.values()) / max(hits.values())
        print(
            f"  --lbs {num_lbs} --fps {fps_per_lb}: {used}/{total_fps} FPs used"
            f"  (min/max load {spread:.2f})"
        )
        assert used == total_fps, f"{num_lbs}x{fps_per_lb} left FPs idle"
        assert spread > 0.5, f"{num_lbs}x{fps_per_lb} badly skewed: {spread:.2f}"

    # ---- end-to-end dispatch through real threads ----------------------
    quiet = io.StringIO()
    with redirect_stdout(quiet):
        fp_queues: list[ThreadSafeQueue[PacketJob]] = [ThreadSafeQueue(1000) for _ in range(4)]
        mgr = LBManager(num_lbs=2, fps_per_lb=2, fp_queues=fp_queues)
        mgr.start_all()

        n = 400
        for i in range(n):
            job = PacketJob(packet_id=i, tuple=tup(i))
            mgr.get_lb_for_packet(job.tuple).get_input_queue().push(job)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if mgr.get_aggregated_stats().total_dispatched >= n:
                break
            time.sleep(0.01)

        mgr.stop_all()
        agg = mgr.get_aggregated_stats()
        drained = [q.size() for q in fp_queues]
        per_lb = [mgr.get_lb(i).get_stats().per_fp_packets for i in range(mgr.get_num_lbs())]

    assert agg.total_received == n, agg
    assert agg.total_dispatched == n, agg
    assert sum(drained) == n, drained
    print(f"\nDispatched {agg.total_dispatched}/{n} packets through 2 LBs x 2 FPs")
    print(f"  per-FP queue depths: {drained}")
    print(f"  per-LB dispatch counts: {per_lb}")
    assert 0 not in drained, f"every FP must now receive traffic, got {drained}"

    # ---- flow affinity: one tuple always lands on the same FP ----------
    lb = LoadBalancer(0, fp_queues, 0)
    t = tup(7)
    assert len({lb.select_fp(t) for _ in range(100)}) == 1, "hashing must be deterministic"
    assert lb.select_fp(t) == lb.select_fp(t.reverse()), "both directions -> same FP"

    print("load_balancer.py self-test OK")
