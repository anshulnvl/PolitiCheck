"""
pipeline/cache.py
=================
Redis-backed result cache for the pipeline orchestrator.

Key  : "pipeline:" + MD5(normalised input text)
Value: JSON-serialised pipeline result dict
TTL  : PIPELINE_CACHE_TTL env var (default 86 400 s / 24 h)

Cache misses and write failures are silent — the pipeline continues without
caching rather than raising an exception.
"""

import hashlib
import json
import os
from typing import Optional

import redis

REDIS_URL  = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL  = int(os.getenv("PIPELINE_CACHE_TTL", 86_400))   # 24 h
_CACHE_DB  = 2
_PREFIX    = "pipeline:"

_client: Optional[redis.Redis] = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(
            f"{REDIS_URL}/{_CACHE_DB}",
            decode_responses=True,
            socket_connect_timeout=2,
        )
    return _client


def _key(text: str) -> str:
    digest = hashlib.md5(text.strip().encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{_PREFIX}{digest}"


def check_cache(text: str) -> Optional[dict]:
    """Return the cached result for *text*, or None on miss / error."""
    try:
        raw = _get_client().get(_key(text))
        return json.loads(raw) if raw else None
    except Exception:
        return None


def write_cache(text: str, result: dict) -> None:
    """Persist *result* under the key for *text*.  Silently drops on error."""
    try:
        _get_client().setex(_key(text), CACHE_TTL, json.dumps(result))
    except Exception:
        pass
