"""Cooperative, cross-thread cancellation for an active agent operation."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar


class OperationCancelled(RuntimeError):
    """Raised at a cooperative cancellation boundary."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = ""

    def cancel(self, reason: str = "Cancelled by user") -> None:
        self.reason = reason
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise OperationCancelled(self.reason or "Operation cancelled")


_CURRENT_TOKEN: ContextVar[CancellationToken | None] = ContextVar(
    "remy_current_cancellation_token", default=None
)


@contextmanager
def bind_cancellation_token(token: CancellationToken):
    marker = _CURRENT_TOKEN.set(token)
    try:
        yield token
    finally:
        _CURRENT_TOKEN.reset(marker)


def current_cancellation_token() -> CancellationToken | None:
    return _CURRENT_TOKEN.get()


def check_cancelled() -> None:
    token = current_cancellation_token()
    if token is not None:
        token.raise_if_cancelled()
