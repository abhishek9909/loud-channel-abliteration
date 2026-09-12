"""Retry wrapper for transient network failures on NDIF remote traces.

A multi-hour remote grid makes thousands of HTTP round-trips (submit, poll,
download); a single DNS hiccup on a compute node ("Name or service not
known") must not kill the run. Traces are stateless — re-executing the whole
with-block is safe — so callers wrap each batch in a zero-arg closure and
pass it here.

Only transient transport errors are retried (matched by class name anywhere
in the exception cause-chain, so httpx need not be imported); server-side
errors (RemoteException: whitelist violations, OOM, bad code) re-raise
immediately.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# httpx / httpcore / stdlib transport-level failures worth retrying
_TRANSIENT = {
    "ConnectError", "ConnectTimeout", "ReadTimeout", "ReadError",
    "WriteError", "PoolTimeout", "ProtocolError", "RemoteProtocolError",
    "ConnectionError", "ConnectionResetError", "BrokenPipeError",
    "gaierror", "timeout", "TimeoutError", "SSLError",
    # truncated result downloads surface at DESERIALIZATION, not transport:
    # nnsight streams the payload and hands incomplete bytes to torch.load ->
    # "EOFError: Ran out of input" / UnpicklingError / zstd errors. The trace
    # is stateless, so re-submitting is safe.
    "EOFError", "UnpicklingError", "ZstdError", "ZstdDecompressionError",
    "IncompleteReadError", "ChunkedEncodingError", "DecompressionBombError",
}


# Server-side RemoteExceptions are NOT retried in general (whitelist
# violations, bad trace code, OOM must fail fast) — EXCEPT when NDIF itself
# says the failure is temporary. Matched on the message text.
_TRANSIENT_MESSAGES = (
    "try again later",
    "error submitting request",
    "temporarily unavailable",
    "service unavailable",
    "too many requests",
)


def is_transient(e: BaseException) -> bool:
    seen = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if any(t.__name__ in _TRANSIENT for t in type(cur).__mro__):
            return True
        msg = str(cur).lower()
        if any(s in msg for s in _TRANSIENT_MESSAGES):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def with_retries(fn: Callable[[], T], *, attempts: int = 7, base_delay: float = 10.0,
                 what: str = "remote trace") -> T:
    # 7 attempts x base 10s doubling -> covers ~10.5 min of deployment
    # unavailability before giving up (checkpointing catches the rest)
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts or not is_transient(e):
                raise
            delay = base_delay * 2 ** (attempt - 1)
            print(f"[remote-retry] {what}: attempt {attempt}/{attempts} failed "
                  f"({type(e).__name__}: {str(e)[:120]}); retrying in {delay:.0f}s")
            time.sleep(delay)
    raise AssertionError("unreachable")
