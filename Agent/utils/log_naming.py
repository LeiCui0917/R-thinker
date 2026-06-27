from __future__ import annotations

import os
import re
import time


def _safe(value: str) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def _abbr_game(game: str | None) -> str:
    g = _safe(game or "")
    if g in ("maze", "mz"):
        return "maze"
    if g in ("chess", "ch"):
        return "chess"
    return "game"


def _abbr_side(side: str | None) -> str:
    s = _safe(side or "")
    if s in ("red", "r"):
        return "red"
    if s in ("blue", "b"):
        return "blue"
    if s in ("white", "w"):
        return "white"
    if s in ("black", "k", "bk"):
        return "black"
    return "unk"


def _abbr_agent(agent: str | None) -> str:
    a = _safe(agent or "")
    table = {
        "think_agent": "think",
        "think": "think",
        "think_without_opp": "think_no_opp",
        "without_opp": "think_no_opp",
        "two": "think_no_opp",
        "think_without_self": "think_no_self",
        "without_self": "think_no_self",
        "tws": "think_no_self",
        "reflexion": "reflexion",
        "cot": "cot",
        "llm_agent": "llm",
        "llm": "llm",
        "memoryllm": "memory",
        "memory": "memory",
        "coding": "coding",
        "rule1": "rule",
        "rule_agent": "rule",
        "random_agent": "random",
        "fast_only": "fast_only",
        "fo": "fast_only",
        "fast": "fast",
        "slow_only": "slow_only",
        "so": "slow_only",
        "slow": "slow",
    }
    if a in table:
        return table[a]
    return (a[:12] if a else "agent")


def _abbr_module(module: str | None) -> str:
    m = _safe(module or "")
    table = {
        "fast": "fast",
        "f": "fast",
        "slow_self": "slow_self",
        "ss": "slow_self",
        "slow_opp": "slow_opp",
        "slow_opponent": "slow_opp",
        "so": "slow_opp",
        "summary": "summary",
        "sm": "summary",
    }
    return table.get(m, (m[:12] if m else ""))


def make_log_seed(
    logs_dir: str | None,
    *,
    run: str | None,
    game: str,
    side: str,
    actor: str | None = None,
    opponent: str | None = None,
    episode: int | None = None,
    retry: int | None = None,
    ext: str = ".log",
) -> str:
    root = os.path.abspath(str(logs_dir or "").strip() or os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(root, exist_ok=True)

    parts = [
        _safe(run or time.strftime("%Y%m%d_%H%M%S")),
        _abbr_game(game),
        _abbr_side(side),
    ]
    if actor:
        parts.append(_safe(actor))
    if opponent:
        parts.append("vs")
        parts.append(_safe(opponent))
    if episode is not None:
        parts.append(f"ep{int(episode)}")
    if retry is not None and int(retry) > 0:
        parts.append(f"retry{int(retry)}")

    return os.path.join(root, "_".join(parts) + str(ext or ".log"))


def make_log_file(
    log_file: str | None,
    *,
    game: str | None = None,
    side: str | None = None,
    agent: str | None = None,
    module: str | None = None,
    k: int | None = None,
    ext: str = ".log",
) -> str:
    hint = str(log_file or "")
    base_dir = os.path.dirname(os.path.abspath(hint)) if hint else ""
    if not base_dir:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(base_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    g = _abbr_game(game)
    s = _abbr_side(side)
    a = _abbr_agent(agent)
    m = _abbr_module(module)

    parts = [ts, g, s, a]
    if m:
        parts.append(m)
    if k is not None:
        parts.append(f"k{int(k)}")

    filename = "_".join(parts) + str(ext or ".log")
    return os.path.join(base_dir, filename)
