"""
Shared helpers for route modules — lazy accessor for api.py module-level state.

All route modules access brain, brain_lock, settings, execute_tool, etc. through
_get_api() so that test patches on remy.web.api.brain take effect.
"""

import asyncio
from fastapi import HTTPException


def _get_api():
    """Lazy accessor — reads from api module (supports test patching)."""
    import remy.web.api as _api

    return _api


# Default timeouts per operation class (seconds)
_TIMEOUT_FAST = 5.0    # simple brain read — count, single record
_TIMEOUT_NORMAL = 10.0  # list, search, store, update
_TIMEOUT_SLOW = 30.0   # bulk import, graph build, consolidation
_TIMEOUT_EVAL = 120.0  # benchmark runs, live validation packs


async def run_in_thread(fn, *args, timeout: float = _TIMEOUT_NORMAL, error_msg: str = "Operation timed out", **kwargs):
    """Run a sync function in a thread with a timeout.

    Raises HTTP 504 if the operation exceeds the timeout.
    All asyncio.to_thread() calls in web routes should use this wrapper.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(fn, *args, **kwargs),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=error_msg)


async def run_lambda_in_thread(fn, timeout: float = _TIMEOUT_NORMAL, error_msg: str = "Operation timed out"):
    """Same as run_in_thread but for lambdas (no positional args unpacking)."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(fn),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=error_msg)
