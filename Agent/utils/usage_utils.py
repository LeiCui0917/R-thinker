"""Small helpers for normalized LLM token usage."""

from __future__ import annotations

from typing import Any


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def normalize_usage(usage: Any) -> dict[str, int]:
    """Normalize a plain usage dict to stable keys."""

    if not isinstance(usage, dict):
        return {}

    prompt_tokens = _to_int(usage.get("prompt_tokens", 0))
    completion_tokens = _to_int(usage.get("completion_tokens", 0))
    total_tokens = _to_int(usage.get("total_tokens"), default=prompt_tokens + completion_tokens)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def sum_usages(usages: list[Any] | tuple[Any, ...]) -> dict[str, int]:
    """Sum normalized usage dicts with stable OpenAI-style keys."""

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for usage in usages or []:
        normalized = normalize_usage(usage)
        prompt_tokens += _to_int(normalized.get("prompt_tokens", 0))
        completion_tokens += _to_int(normalized.get("completion_tokens", 0))
        total_tokens += _to_int(normalized.get("total_tokens", 0))

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def add_total_tokens(current_total: Any, usage: Any) -> int:
    """Add usage.total_tokens to running total safely."""

    return _to_int(current_total, 0) + _to_int(normalize_usage(usage).get("total_tokens", 0), 0)


def add_wrapper_tokens_from_inner_total(
    current_total: Any,
    last_inner_total: Any,
    inner_total: Any,
    usage: Any = None,
) -> tuple[int, int]:
    """Accumulate wrapper token totals using inner total-token deltas.

    Returns `(new_total, new_last_inner_total)`.
    - Primary path: add delta of `inner_total` vs `last_inner_total`.
    - Fallback path: if totals are unavailable, add normalized `usage.total_tokens`.
    """

    curr = _to_int(current_total, 0)
    prev_inner = _to_int(last_inner_total, 0)
    curr_inner = _to_int(inner_total, -1)

    if curr_inner >= 0:
        if curr_inner >= prev_inner:
            delta = curr_inner - prev_inner
        else:
            delta = curr_inner
        return curr + max(0, delta), curr_inner

    return add_total_tokens(curr, usage), prev_inner
