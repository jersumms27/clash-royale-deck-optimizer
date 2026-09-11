"""Thin client for the official Clash Royale API (stdlib only).

Mirrors optimizer/cr_api.py (urllib + Bearer token) and adds what a crawler
needs: a thread-safe rate limiter, retries with backoff, and error classes
that let the crawl loop tell "skip this player" from "stop everything".
"""

from __future__ import annotations

import gzip
import json
import random
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from optimizer import config

# RoyaleAPI's proxy lets an IP-locked token work from anywhere: whitelist this
# IP when creating the token, then set CR_API_BASE to the proxy URL.
PROXY_BASE = "https://proxy.royaleapi.dev/v1"
PROXY_IP = "45.79.218.79"

NO_TOKEN_MESSAGE = (
    "No API token. Get one at https://developer.clashroyale.com, then set "
    "CR_API_TOKEN or put it in token.txt in the project root."
)
FORBIDDEN_MESSAGE = (
    "HTTP {code}: the API rejected the token. Tokens are IP-locked, so this usually "
    "means your current IP is not on the token's whitelist. Either re-create the "
    f"token for your current IP, or create one whitelisted for {PROXY_IP} and run "
    f"with CR_API_BASE={PROXY_BASE}."
)

# Supercell tags use the alphabet 0289PYLQGRJCUV, but we accept any
# alphanumerics so a surprise never silently drops a player (O -> 0 is the
# one classic typo worth fixing).
TAG_CHARS = frozenset("0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ")


class ApiError(RuntimeError):
    """A request failed after retries. `status` is the HTTP code, if any."""

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class NotFound(ApiError):
    """HTTP 404: unknown tag / no ranking for this location. Never retried."""


class FatalApiError(ApiError):
    """HTTP 401/403: the token itself is the problem. Abort the whole crawl."""


# --------------------------------------------------------------------------- #
# Tags                                                                         #
# --------------------------------------------------------------------------- #
def norm_tag(tag: str) -> str:
    """Canonical '#UPPER' form. Accepts a missing '#', lowercase, and O-for-0."""
    s = str(tag).strip().upper().lstrip("#").replace("O", "0")
    if not s or not set(s) <= TAG_CHARS:
        raise ValueError(f"not a valid player tag: {tag!r}")
    return "#" + s


def bare_tag(tag: str) -> str:
    """'#ABC' -> 'ABC' (used inside battle ids and file names)."""
    return norm_tag(tag)[1:]


def tag_path(tag: str) -> str:
    """URL path segment for a tag: '#ABC' -> '%23ABC'."""
    return urllib.parse.quote(norm_tag(tag), safe="")


# --------------------------------------------------------------------------- #
# Rate limiting                                                                #
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Token bucket shared by all worker threads.

    A caller reserves a token under the lock (the bucket may go negative) and
    sleeps *outside* the lock, so waiting threads don't serialise each other.
    `penalize` pushes a global not-before time, which is how one worker's 429
    pauses every worker.
    """

    def __init__(self, rate: float, burst: int = 1):
        if rate <= 0:
            raise ValueError("rate must be > 0 requests/second")
        self.rate = float(rate)
        self.burst = max(1, int(burst))
        self._tokens = float(self.burst)
        self._last = time.monotonic()
        self._not_before = 0.0
        self._lock = threading.Lock()

    def acquire(self, stop: threading.Event | None = None) -> None:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
            self._last = now
            wait = max(0.0, self._not_before - now)
            if self._tokens < 1.0:
                wait += (1.0 - self._tokens) / self.rate
            self._tokens -= 1.0
        if wait > 0:
            _sleep(wait, stop)

    def penalize(self, seconds: float) -> None:
        with self._lock:
            self._not_before = max(self._not_before, time.monotonic() + max(0.0, seconds))


def _sleep(seconds: float, stop: threading.Event | None) -> None:
    """Sleep that returns early once `stop` is set (keeps shutdown snappy)."""
    if stop is None:
        time.sleep(seconds)
    else:
        stop.wait(seconds)


# --------------------------------------------------------------------------- #
# Client                                                                       #
# --------------------------------------------------------------------------- #
class Client:
    def __init__(
        self,
        token: str,
        base: str = config.CR_API_BASE,
        limiter: RateLimiter | None = None,
        timeout: float = 20.0,
        max_retries: int = 4,
        stop: threading.Event | None = None,
    ):
        self.token = token
        self.base = base.rstrip("/")
        self.limiter = limiter
        self.timeout = timeout
        self.max_retries = max_retries
        self.stop = stop
        self.requests = 0  # total attempts actually sent (for req/s reporting)
        self._count_lock = threading.Lock()

    # -- low level ---------------------------------------------------------- #
    def get(self, path: str, params: dict | None = None) -> Any:
        """GET `path` (relative to the base URL) and return the parsed JSON.

        Retries 429 (honouring Retry-After), 5xx and network errors with
        exponential backoff. Raises NotFound on 404 and FatalApiError on
        401/403; anything else that keeps failing raises ApiError.
        """
        url = f"{self.base}/{path.lstrip('/')}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
            },
        )

        attempt = 0
        while True:
            if self.stop is not None and self.stop.is_set():
                raise ApiError("stopped")
            if self.limiter is not None:
                self.limiter.acquire(self.stop)
            with self._count_lock:
                self.requests += 1
            attempt += 1
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                        raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "ignore")
                code = exc.code
                if code == 404:
                    raise NotFound(f"HTTP 404 for {path}", code, body) from exc
                if code in (401, 403):
                    raise FatalApiError(FORBIDDEN_MESSAGE.format(code=code), code, body) from exc
                if code == 429:
                    delay = _retry_after(exc.headers.get("Retry-After"), attempt)
                    if self.limiter is not None:
                        self.limiter.penalize(delay)
                    else:
                        _sleep(delay, self.stop)
                    if attempt > self.max_retries:
                        raise ApiError(f"HTTP 429 for {path} after {attempt} attempts", code, body) from exc
                    continue
                if 500 <= code < 600 and attempt <= self.max_retries:
                    _sleep(_backoff(attempt), self.stop)
                    continue
                raise ApiError(f"HTTP {code} for {path}: {body[:200]}", code, body) from exc
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
                if attempt <= self.max_retries:
                    _sleep(_backoff(attempt), self.stop)
                    continue
                raise ApiError(f"could not reach the API for {path}: {exc}") from exc
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ApiError(f"non-JSON response for {path}: {exc}") from exc

    # -- endpoints ---------------------------------------------------------- #
    def battlelog(self, tag: str) -> list[dict]:
        """The player's ~25 most recent battles (a bare JSON array)."""
        data = self.get(f"players/{tag_path(tag)}/battlelog")
        if isinstance(data, dict):  # be lenient if a proxy wraps it
            data = data.get("items", [])
        return list(data)

    def locations(self) -> list[dict]:
        return list(self.get("locations", {"limit": 1000}).get("items", []))

    def pol_players(self, location: int | str = "global", limit: int = 200) -> list[str]:
        """Path of Legend ranking for a location -> normalised player tags."""
        data = self.get(f"locations/{location}/pathoflegend/players", {"limit": limit})
        return _tags_of(data.get("items", []))


def _tags_of(items: list[dict]) -> list[str]:
    tags = []
    for it in items:
        try:
            tags.append(norm_tag(it["tag"]))
        except (KeyError, ValueError):
            continue
    return tags


def _retry_after(header: str | None, attempt: int) -> float:
    try:
        delay = float(header) if header else 0.0
    except ValueError:
        delay = 0.0
    if delay <= 0:
        delay = float(2 ** min(attempt, 6))
    return min(delay, 120.0)


def _backoff(attempt: int) -> float:
    return min(30.0, 2 ** min(attempt, 5) + random.random())


def make_client(
    rps: float = 5.0,
    burst: int = 5,
    timeout: float = 20.0,
    max_retries: int = 4,
    stop: threading.Event | None = None,
    base: str | None = None,
) -> Client:
    """Build a Client from config (token + base URL). Raises if there's no token."""
    token = config.get_api_token()
    if not token:
        raise RuntimeError(NO_TOKEN_MESSAGE)
    return Client(
        token,
        base=base or config.CR_API_BASE,
        limiter=RateLimiter(rps, burst),
        timeout=timeout,
        max_retries=max_retries,
        stop=stop,
    )
