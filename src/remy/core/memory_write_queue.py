"""Single-lane executor for background Aura memory mutations."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable


class MemoryWriteQueue:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="remy-memory-write")
        self._lock = threading.Lock()
        self._pending: set[Future] = set()
        self._closed = False

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("Background memory write queue is closed")
            future = self._executor.submit(fn, *args, **kwargs)
            self._pending.add(future)
        future.add_done_callback(self._discard)
        return future

    def _discard(self, future: Future) -> None:
        with self._lock:
            self._pending.discard(future)

    async def run(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        return await asyncio.wrap_future(self.submit(fn, *args, **kwargs))

    def drain(self, timeout: float = 15.0) -> bool:
        with self._lock:
            pending = set(self._pending)
        if not pending:
            return True
        _done, remaining = wait(pending, timeout=timeout)
        return not remaining

    def close(self, timeout: float = 15.0) -> bool:
        drained = self.drain(timeout)
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=False)
        return drained


_QUEUE = MemoryWriteQueue()
_QUEUE_LOCK = threading.Lock()


def get_memory_write_queue() -> MemoryWriteQueue:
    global _QUEUE
    with _QUEUE_LOCK:
        if _QUEUE._closed:
            _QUEUE = MemoryWriteQueue()
    return _QUEUE
