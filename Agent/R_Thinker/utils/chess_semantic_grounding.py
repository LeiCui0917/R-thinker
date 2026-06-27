"""Chess 语义文本 grounding。

作用：
1. 把语义树节点、路径描述中的抽象坐标占位表达落地为真实棋盘格。
2. 在给定 FEN 的前提下，把相对 king 的描述转换成具体方格或方格集合。

当前支持的典型形式：
- `(x, n)`
- `(x+1, n-2)`
- `(x±1, n±2)`
- `(x+Δx, n)`
- `(x+|Δx|, n-|Δn|)`
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple


# ============================================================
# 模块一：常量
# 说明：集中放置坐标轴、符号兼容和正则模板。
# ============================================================

FILES = "abcdefgh"

# 一些模型会把 ± 输出成乱码字符，或者写成 +/-，这里统一兼容。
_PM_TOKEN_RE = r"(?:å¤|\x1b|\+/-)"

# 只匹配抽象坐标对块，如：
# - (x+1, n-2)
# - (x±1, n±2)
# - (x+Δx, n)
#
# 外层普通文本不由这个正则处理，这样整句路径/节点描述可以原样保留。
_PAIR_RE = re.compile(
    rf"\(\s*(?P<xexpr>x\s*(?:[+-]\s*(?:\d+|\|?\s*[èž–æœª\?]\s*[a-z]\s*\|?|abs\s*\(\s*[èž–æœª\?]\s*[a-z]\s*\))|{_PM_TOKEN_RE}\s*\d+)?)\s*,\s*(?P<nexpr>[ny]\s*(?:[+-]\s*(?:\d+|\|?\s*[èž–æœª\?]\s*[a-z]\s*\|?|abs\s*\(\s*[èž–æœª\?]\s*[a-z]\s*\))|{_PM_TOKEN_RE}\s*\d+)?)\s*\)",
    flags=re.IGNORECASE,
)


# ============================================================
# 模块二：坐标基础层
# 说明：负责方格字符串与内部坐标表示之间的转换。
# ============================================================

def _square_to_coords(square: str) -> Optional[Tuple[int, int]]:
    """把 `e4` 这类方格转成 `(file_idx, rank_num)`。"""
    if not square or len(square) < 2:
        return None
    file_idx = FILES.find(square[0].lower())
    try:
        rank_num = int(square[1])
    except Exception:
        return None
    if file_idx < 0 or rank_num < 1 or rank_num > 8:
        return None
    return file_idx, rank_num


def _coords_to_square(file_idx: int, rank_num: int) -> Optional[str]:
    """把内部坐标转回 `e4` 这类方格；越界时返回 `None`。"""
    if file_idx < 0 or file_idx > 7 or rank_num < 1 or rank_num > 8:
        return None
    return f"{FILES[int(file_idx)]}{int(rank_num)}"


def _format_square_set(squares: list[str]) -> Optional[str]:
    """把候选格子格式化成单点或集合文本。"""
    if not squares:
        return None
    uniq = sorted(set(squares), key=lambda s: (int(s[1]), FILES.find(s[0].lower())))
    if len(uniq) == 1:
        return uniq[0]
    return "{" + ", ".join(uniq) + "}"


# ============================================================
# 模块三：锚点提取层
# 说明：先从文本和 FEN 中找出 grounding 的参考 king。
# ============================================================

def _detect_king_color_from_text(text: str) -> Optional[str]:
    """从文本中识别 `White King` 或 `Black King`。"""
    if not text:
        return None
    if re.search(r"\bWhite\s*King\b|\bWhite_king\b", text, flags=re.IGNORECASE):
        return "w"
    if re.search(r"\bBlack\s*King\b|\bBlack_king\b", text, flags=re.IGNORECASE):
        return "b"
    return None


def _detect_piece_type_from_text(text: str) -> Optional[str]:
    """尽量识别文本里提到的棋子类型，用于后续约束射线展开。"""
    if not text:
        return None
    m = re.search(
        r"\b(?:white|black)\s+(pawn|knight|bishop|rook|queen|king)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    piece_type = str(m.group(1) or "").strip().lower()
    return piece_type or None


def _fen_find_king_square(fen: str, color: str) -> Optional[str]:
    """只扫描 FEN 棋盘段，找出指定颜色 king 的真实方格。"""
    try:
        board_part = (fen or "").strip().split()[0]
        ranks = board_part.split("/")
        if len(ranks) != 8:
            return None

        target = "K" if color == "w" else "k"
        for r_index, row in enumerate(ranks):
            file_idx = 0
            for ch in row:
                if ch.isdigit():
                    file_idx += int(ch)
                    continue
                if file_idx > 7:
                    break
                if ch == target:
                    return f"{FILES[file_idx]}{8 - r_index}"
                file_idx += 1
        return None
    except Exception:
        return None


# ============================================================
# 模块四：占位表达解析层
# 说明：把 x+1 / x±2 / x+Δx 这类轴表达统一解析成结构。
# ============================================================

@dataclass(frozen=True)
class _AxisSpec:
    """单个轴表达的统一表示。

    - fixed: 固定偏移，例如 `x+2`、`n-1`
    - pm:    有限 ± 集合，例如 `x±1`、`n±2`
    - delta: 射线集合，例如 `x+Δx`、`n-|Δn|`
      - `None`：不是 delta
      - `0`：双向
      - `+1/-1`：单向
    """

    fixed: Optional[int] = None
    pm: Optional[int] = None
    delta: Optional[int] = None


def _parse_axis_expr(expr: str, axis_char: str) -> _AxisSpec:
    """解析单个坐标轴表达，如 `x`、`x+2`、`x±1`、`x+Δx`。"""
    expr = (expr or "").replace(" ", "")
    if not expr:
        return _AxisSpec()

    if re.fullmatch(rf"{axis_char}", expr, flags=re.IGNORECASE):
        return _AxisSpec(fixed=0)

    m = re.fullmatch(rf"{axis_char}([+-])(\d+)", expr, flags=re.IGNORECASE)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        return _AxisSpec(fixed=sign * int(m.group(2)))

    m = re.fullmatch(rf"{axis_char}{_PM_TOKEN_RE}(\d+)", expr, flags=re.IGNORECASE)
    if m:
        return _AxisSpec(fixed=0, pm=int(m.group(1)))

    m = re.fullmatch(rf"{axis_char}([+-])\|([èž–æœª\?])[a-z]\|", expr, flags=re.IGNORECASE)
    if m:
        return _AxisSpec(fixed=0, delta=(1 if m.group(1) == "+" else -1))

    m = re.fullmatch(rf"{axis_char}([+-])abs\(([èž–æœª\?])[a-z]\)", expr, flags=re.IGNORECASE)
    if m:
        return _AxisSpec(fixed=0, delta=(1 if m.group(1) == "+" else -1))

    m = re.fullmatch(rf"{axis_char}([+-])([èž–æœª\?])[a-z]", expr, flags=re.IGNORECASE)
    if m:
        return _AxisSpec(fixed=0, delta=0)

    return _AxisSpec()


# ============================================================
# 模块五：语义展开层
# 说明：把解析后的抽象表达，结合 king 方格，展开成具体方格集合。
# ============================================================

def _expand_delta_relative_to_anchor(
    anchor_sq: str,
    dx_spec: _AxisSpec,
    dn_spec: _AxisSpec,
    *,
    piece_type: Optional[str] = None,
) -> Optional[str]:
    """把 delta/ray 形式展开成具体格子集合。"""
    if dx_spec.delta is None and dn_spec.delta is None:
        return None
    if dx_spec.fixed is None or dn_spec.fixed is None:
        return None

    anchor = _square_to_coords(anchor_sq)
    if not anchor:
        return None
    anchor_file, anchor_rank = anchor
    base_file = anchor_file + int(dx_spec.fixed)
    base_rank = anchor_rank + int(dn_spec.fixed)

    def _dirs(v: int) -> Tuple[int, ...]:
        return (1, -1) if int(v) == 0 else (int(v),)

    squares: list[str] = []
    piece_type = (piece_type or "").strip().lower()

    if dx_spec.delta is not None and dn_spec.delta is not None:
        if piece_type and piece_type not in {"bishop", "queen"}:
            return None
        for step_file in _dirs(int(dx_spec.delta)):
            for step_rank in _dirs(int(dn_spec.delta)):
                for t in range(1, 8):
                    sq = _coords_to_square(base_file + step_file * t, base_rank + step_rank * t)
                    if sq and sq != anchor_sq:
                        squares.append(sq)
        return _format_square_set(squares)

    if dx_spec.delta is not None:
        if piece_type and piece_type not in {"rook", "queen"}:
            return None
        for step_file in _dirs(int(dx_spec.delta)):
            for t in range(1, 8):
                sq = _coords_to_square(base_file + step_file * t, base_rank)
                if sq and sq != anchor_sq:
                    squares.append(sq)
        return _format_square_set(squares)

    if dn_spec.delta is not None:
        if piece_type and piece_type not in {"rook", "queen"}:
            return None
        for step_rank in _dirs(int(dn_spec.delta)):
            for t in range(1, 8):
                sq = _coords_to_square(base_file, base_rank + step_rank * t)
                if sq and sq != anchor_sq:
                    squares.append(sq)
        return _format_square_set(squares)

    return None


# ============================================================
# 模块六：顶层文本 grounding 入口
# 说明：驱动整套流程，把整句文本中的抽象坐标块替换掉。
# ============================================================

def ground_chess_semantic_text(fen: str, text: str) -> str:
    """把语义文本中的抽象坐标对落地成真实棋盘格。"""
    if not text:
        return text

    if "\n" in str(text):
        lines = str(text).splitlines()
        return "\n".join(ground_chess_semantic_text(fen, line) if line.strip() else line for line in lines)

    text = re.sub(r"\bat\s+abstract\s+coordinate\b", "at", text, flags=re.IGNORECASE)

    # 第一步：明确找到锚点 king。没有写清楚，就不做隐式推断。
    king_color = _detect_king_color_from_text(text)
    if king_color is None:
        return text

    king_sq = _fen_find_king_square(fen, king_color)
    if not king_sq:
        return text

    piece_type = _detect_piece_type_from_text(text)

    # 先把 `(x,y)` 这种“锚点自身”直接替换成 king 方格。
    out = re.sub(r"\(\s*x\s*,\s*y\s*\)", king_sq, text, flags=re.IGNORECASE)

    anchor = _square_to_coords(king_sq)
    if not anchor:
        return text
    anchor_file, anchor_rank = anchor

    # 第二步：逐个抽象坐标对块做 grounding。
    def _pair_repl(m: re.Match) -> str:
        xexpr = m.group("xexpr")
        nexpr = m.group("nexpr")

        dx = _parse_axis_expr(xexpr, "x")
        axis2 = "y" if str(nexpr).strip().lower().startswith("y") else "n"
        dn = _parse_axis_expr(nexpr, axis2)
        if dx.fixed is None or dn.fixed is None:
            return m.group(0)

        # 先处理 delta/ray：它们对应一条线或一个集合，而不是单点。
        delta_expanded = _expand_delta_relative_to_anchor(king_sq, dx, dn, piece_type=piece_type)
        if delta_expanded:
            return delta_expanded

        # 再处理有限 ± 集合。
        if dx.pm is not None and dn.pm is not None:
            candidates: list[str] = []
            for sx in (1, -1):
                for sn in (1, -1):
                    sq = _coords_to_square(
                        anchor_file + sx * int(dx.pm),
                        anchor_rank + sn * int(dn.pm),
                    )
                    if sq and sq != king_sq:
                        candidates.append(sq)
            return _format_square_set(candidates) or "{}"

        if dx.pm is not None:
            candidates: list[str] = []
            for sx in (1, -1):
                sq = _coords_to_square(
                    anchor_file + sx * int(dx.pm),
                    anchor_rank + int(dn.fixed),
                )
                if sq and sq != king_sq:
                    candidates.append(sq)
            return _format_square_set(candidates) or "{}"

        if dn.pm is not None:
            candidates: list[str] = []
            for sn in (1, -1):
                sq = _coords_to_square(
                    anchor_file + int(dx.fixed),
                    anchor_rank + sn * int(dn.pm),
                )
                if sq and sq != king_sq:
                    candidates.append(sq)
            return _format_square_set(candidates) or "{}"

        # 最后处理纯固定偏移单点。
        if int(dx.fixed) == 0 and int(dn.fixed) == 0:
            return king_sq

        sq = _coords_to_square(
            anchor_file + int(dx.fixed),
            anchor_rank + int(dn.fixed),
        )
        return sq or "{}"

    return _PAIR_RE.sub(_pair_repl, out)
