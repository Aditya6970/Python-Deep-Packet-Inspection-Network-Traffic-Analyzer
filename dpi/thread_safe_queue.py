"""Bounded blocking queue for passing packets between threads.

Python port of ``include/thread_safe_queue.h`` (C++ ``namespace DPI``).

Used for Reader -> LB -> FP communication.  Each stage owns an input queue; a
producer blocks when the consumer falls behind, which is what applies
back-pressure through the pipeline instead of letting memory grow without
bound.

C++ concepts replaced
---------------------
``template<typename T> class ThreadSafeQueue``
    Becomes ``ThreadSafeQueue(Generic[T])`` — the same static type information,
    erased at runtime as C++ templates are erased at compile time.

``std::mutex`` + two ``std::condition_variable``s
    Becomes one :class:`threading.Lock` shared by two
    :class:`threading.Condition` objects, which is the direct analogue: both
    condvars must wait on the *same* mutex, and passing the lock to both
    ``Condition`` constructors guarantees that.

``cv.wait(lock, predicate)``
    Becomes ``cond.wait_for(predicate)``, which likewise loops internally and
    is immune to spurious wakeups.

``std::optional<T> pop()``
    Becomes ``T | None``.  Note this makes ``None`` unusable as a queue item —
    acceptable here, since the queue only ever carries ``PacketJob`` objects.

Why not :class:`queue.Queue`?
    ``queue.Queue`` has no shutdown that wakes blocked producers *and*
    consumers while still allowing the backlog to drain, and its ``put`` raises
    :class:`queue.Full` rather than silently discarding.  Reproducing the C++
    semantics on top of it would take more code than implementing them
    directly, and would obscure the shutdown behaviour documented below.
"""

from __future__ import annotations

import threading
from typing import Final, Generic, TypeVar

__all__ = ["DEFAULT_MAX_SIZE", "ThreadSafeQueue"]

T = TypeVar("T")

#: Default queue capacity, matching the C++ default argument.
DEFAULT_MAX_SIZE: Final[int] = 10000


class ThreadSafeQueue(Generic[T]):
    """Thread-safe bounded queue with shutdown support.

    Mirrors ``DPI::ThreadSafeQueue<T>``.

    Shutdown semantics (preserved exactly, and easy to get wrong):

    * :meth:`push` **silently discards** the item if shutdown is signalled
      while it is blocked waiting for space.  It does not raise and does not
      report the loss — the C++ ``if (shutdown_) return;`` drops the item on
      the floor.  Callers cannot tell a dropped packet from an enqueued one.
    * :meth:`pop` keeps returning **buffered items after shutdown** and only
      reports exhaustion once the backlog is empty.  So a consumer draining
      after shutdown still sees every item that made it in.
    * Shutdown is one-way; there is no reset, as in the original.
    """

    __slots__ = ("_queue", "_lock", "_not_empty", "_not_full", "_max_size", "_shutdown")

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE) -> None:
        from collections import deque

        self._queue: deque[T] = deque()
        # One mutex, two condition variables — exactly the C++ arrangement.
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._max_size = max_size
        self._shutdown = False

    # ------------------------------------------------------------------
    # Producing
    # ------------------------------------------------------------------
    def push(self, item: T) -> None:
        """Push an item, blocking while the queue is full.

        Mirrors ``void push(T item)``.

        WARNING: if shutdown is signalled while blocked, the item is
        **silently dropped** — see the class docstring.
        """
        with self._not_full:
            self._not_full.wait_for(lambda: len(self._queue) < self._max_size or self._shutdown)

            if self._shutdown:
                return  # C++: `if (shutdown_) return;` — item is discarded

            self._queue.append(item)
            self._not_empty.notify()

    def try_push(self, item: T) -> bool:
        """Push without blocking; return ``False`` if full or shut down.

        Mirrors ``bool tryPush(T item)``.
        """
        with self._lock:
            if len(self._queue) >= self._max_size or self._shutdown:
                return False
            self._queue.append(item)
            self._not_empty.notify()
            return True

    # ------------------------------------------------------------------
    # Consuming
    # ------------------------------------------------------------------
    def pop(self) -> T | None:
        """Pop an item, blocking while the queue is empty.

        Mirrors ``std::optional<T> pop()``.  Returns ``None`` only when the
        queue is both shut down *and* drained — buffered items are still
        delivered after shutdown.
        """
        with self._not_empty:
            self._not_empty.wait_for(lambda: len(self._queue) > 0 or self._shutdown)

            if not self._queue:
                return None

            item = self._queue.popleft()
            self._not_full.notify()
            return item

    def pop_with_timeout(self, timeout_ms: float) -> T | None:
        """Pop an item, giving up after ``timeout_ms`` milliseconds.

        Mirrors ``std::optional<T> popWithTimeout(std::chrono::milliseconds)``.
        The argument is milliseconds, as in the C++; :mod:`threading` works in
        seconds, so it is converted here.

        Returns ``None`` on timeout, and also when shutdown leaves the queue
        empty — the original conflates those two cases too.
        """
        with self._not_empty:
            if not self._not_empty.wait_for(
                lambda: len(self._queue) > 0 or self._shutdown,
                timeout=timeout_ms / 1000.0,
            ):
                return None  # Timeout

            if not self._queue:
                return None

            item = self._queue.popleft()
            self._not_full.notify()
            return item

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def empty(self) -> bool:
        """Return whether the queue is currently empty (``empty()``)."""
        with self._lock:
            return not self._queue

    def size(self) -> int:
        """Return the current number of queued items (``size()``)."""
        with self._lock:
            return len(self._queue)

    def max_size(self) -> int:
        """Return the configured capacity."""
        return self._max_size

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Signal shutdown and wake every waiting thread.

        Mirrors ``void shutdown()``.  Idempotent and one-way.
        """
        with self._lock:
            self._shutdown = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def is_shutdown(self) -> bool:
        """Return whether shutdown has been signalled (``isShutdown()``)."""
        with self._lock:
            return self._shutdown

    # ------------------------------------------------------------------
    # Pythonic extras (no C++ counterpart, additive only)
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.size()

    def __repr__(self) -> str:
        return (
            f"ThreadSafeQueue(size={self.size()}/{self._max_size}, "
            f"shutdown={self.is_shutdown()})"
        )

    # C++-style aliases
    tryPush = try_push
    popWithTimeout = pop_with_timeout
    isShutdown = is_shutdown


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import time

    # --- basic FIFO ordering -------------------------------------------
    q: ThreadSafeQueue[int] = ThreadSafeQueue(max_size=4)
    for i in range(4):
        assert q.try_push(i)
    assert q.size() == 4 and not q.empty()
    assert not q.try_push(99), "try_push must refuse a full queue"
    assert [q.pop() for _ in range(4)] == [0, 1, 2, 3]
    assert q.empty()

    # --- pop_with_timeout returns None rather than hanging --------------
    t0 = time.monotonic()
    assert q.pop_with_timeout(50) is None
    assert 0.03 <= time.monotonic() - t0 < 0.5

    # --- producers block on a full queue, then drain --------------------
    q2: ThreadSafeQueue[int] = ThreadSafeQueue(max_size=2)
    produced: list[int] = []

    def producer() -> None:
        for i in range(10):
            q2.push(i)
            produced.append(i)

    th = threading.Thread(target=producer)
    th.start()
    time.sleep(0.05)
    assert len(produced) < 10, "producer should be blocked by the bound"
    got = [q2.pop() for _ in range(10)]
    th.join()
    assert got == list(range(10))

    # --- shutdown drains the backlog before reporting exhaustion --------
    q3: ThreadSafeQueue[int] = ThreadSafeQueue(max_size=10)
    for i in range(3):
        q3.push(i)
    q3.shutdown()
    assert q3.is_shutdown()
    assert [q3.pop() for _ in range(3)] == [0, 1, 2], "buffered items survive shutdown"
    assert q3.pop() is None, "exhausted + shutdown -> None"
    assert not q3.try_push(1), "try_push refused after shutdown"

    # --- shutdown wakes a blocked consumer ------------------------------
    q4: ThreadSafeQueue[int] = ThreadSafeQueue()
    result: list[int | None] = []
    tc = threading.Thread(target=lambda: result.append(q4.pop()))
    tc.start()
    time.sleep(0.05)
    q4.shutdown()
    tc.join(timeout=2)
    assert not tc.is_alive(), "shutdown must wake a blocked pop()"
    assert result == [None]

    # --- shutdown silently discards a blocked push (documented quirk) ---
    q5: ThreadSafeQueue[int] = ThreadSafeQueue(max_size=1)
    q5.push(1)
    tp = threading.Thread(target=lambda: q5.push(2))
    tp.start()
    time.sleep(0.05)
    q5.shutdown()
    tp.join(timeout=2)
    assert not tp.is_alive(), "shutdown must wake a blocked push()"
    assert q5.pop() == 1
    assert q5.pop() is None, "the blocked item 2 was silently dropped"

    # --- many producers / many consumers, nothing lost or duplicated ----
    q6: ThreadSafeQueue[int] = ThreadSafeQueue(max_size=16)
    n_items, n_prod, n_cons = 500, 4, 4
    seen: list[int] = []
    seen_lock = threading.Lock()

    def prod(base: int) -> None:
        for i in range(n_items):
            q6.push(base * n_items + i)

    def cons() -> None:
        while True:
            item = q6.pop_with_timeout(200)
            if item is None:
                return
            with seen_lock:
                seen.append(item)

    ps = [threading.Thread(target=prod, args=(b,)) for b in range(n_prod)]
    cs = [threading.Thread(target=cons) for _ in range(n_cons)]
    for t in ps + cs:
        t.start()
    for t in ps:
        t.join()
    for t in cs:
        t.join()
    assert sorted(seen) == list(range(n_prod * n_items)), f"lost/dup: {len(seen)}"
    print(f"4x4 stress: {len(seen)} items, none lost or duplicated")

    print("thread_safe_queue.py self-test OK")
