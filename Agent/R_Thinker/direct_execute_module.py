"""Direct Execute 规则执行模块。

定位：
- 该模块在不调用 LLM 的前提下，根据语义规则直接尝试生成可执行动作。
- 常用于 Think/Coding/SlowOnly 的“规则命中即直连执行”路径。

主要能力：
1) Chess：从 subgoal 文本中识别棋子/位置/王位置并推断强制吃王 UCI。
2) Maze：从 subgoal_path 中识别 `(x,y)->(x2,y2)`，若起点匹配当前位置则直接走下一步。

文件组织：
- Part A: Chess 直接执行
- Part B: Maze 直接执行
"""

from __future__ import annotations

import json
import re
from typing import Iterable, Optional, Tuple, Any

import chess


# ============================================================
# Part A: Chess 直接执行
# ============================================================

_SQUARE_RE = r"[a-h][1-8]"

_PIECE_NAME_TO_TYPE = {
    "pawn": chess.PAWN,
    "knight": chess.KNIGHT,
    "bishop": chess.BISHOP,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
    "king": chess.KING,
}


def pick_forced_uci_from_subgoal(
    state_str: str,
    subgoal_text: str,
    legal_moves: Iterable[str],
    player: str,
) -> Optional[str]:
    """规则接手：若子目标描述了“己方某棋子在某格”，且该棋子能立刻吃到对方王，则直接返回该 UCI。

    触发条件（与你的两条规则对齐）：
    1) 目标棋子确实在语义描述的 src 格子（颜色+类型一致）
    2) src -> 对方王格子 的走法在 legal_moves 中存在（含兵升变 UCI）

    否则返回 None，让 LLM 决策。

        支持的语义模式（大小写不敏感）：
      'Black Bishop at d2 relative to White King e1 ...'
      'Black Bishop at {d2, f2, ...} relative to White King e1 ...'
      'White Queen at h5 relative to Black King e8 ...'
    """
    p = (player or "").strip().lower()
    if p not in {"w", "b"}:
        return None

    my_color = "White" if p == "w" else "Black"
    opp_color = "Black" if p == "w" else "White"

    try:
        fen = (state_str or "").split("|", 1)[0]
        if " * " in fen:
            fen = fen.replace(" * ", " w ")
        board = chess.Board(fen)
    except Exception:
        return None

    # True opponent king square from the board.
    opp_king_square = board.king(chess.BLACK if p == "w" else chess.WHITE)
    if opp_king_square is None:
        return None
    opp_king_sq = chess.square_name(opp_king_square).lower()

    # Parse: "<MyColor> <Piece> at <src> relative to <OppColor> King <...>"
    m = re.search(
        rf"{my_color}\s+(?P<piece>Pawn|Knight|Bishop|Rook|Queen|King)\s+at\s+(?P<src>\{{[^}}]+\}}|{_SQUARE_RE})\s+relative\s+to\s+{opp_color}\s+King\s+(?P<king>{_SQUARE_RE})",
        subgoal_text or "",
        flags=re.IGNORECASE,
    )
    if not m:
        return None

    piece_name = (m.group("piece") or "").strip().lower()
    src_text = (m.group("src") or "").strip()
    if not src_text:
        return None
    if src_text.startswith("{") and src_text.endswith("}"):
        inner = src_text[1:-1].strip()
        if not inner:
            return None
        src_squares = [sq for sq in (p.strip().lower() for p in inner.split(",")) if re.fullmatch(_SQUARE_RE, sq)]
    else:
        src_squares = [src_text.lower()] if re.fullmatch(_SQUARE_RE, src_text.lower()) else []
    if not src_squares:
        return None

    # If the semantic king square disagrees with board reality, don't take over.
    king_sq_in_text = (m.group("king") or "").strip().lower()
    if king_sq_in_text and king_sq_in_text != opp_king_sq:
        return None

    piece_type = _PIECE_NAME_TO_TYPE.get(piece_name)
    if piece_type is None:
        return None

    my_is_white = my_color.lower() == "white"
    legal = [str(mv).strip().lower() for mv in (legal_moves or [])]
    legal_set = set(legal)

    for src in src_squares:
        # 1) Verify the target piece is actually on src (type+color).
        try:
            src_piece = board.piece_at(chess.parse_square(src))
        except Exception:
            continue
        if not src_piece:
            continue
        if src_piece.piece_type != piece_type:
            continue
        if bool(src_piece.color) != bool(my_is_white):
            continue

        # 2) Verify a legal capture move exists from src to opponent king square.
        base = f"{src}{opp_king_sq}"
        if base in legal_set:
            return base
        # Promotion UCIs: e7e8q, etc.
        for mv in legal:
            if mv.startswith(base):
                return mv

    return None


# ============================================================
# Part B: Maze 直接执行
# ============================================================

_MAZE_STEP_RE = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*to\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)",
    flags=re.IGNORECASE,
)


def pick_forced_maze_target_from_subgoal_path(
    state: Any,
    player: str,
    subgoal_path: str,
    legal_moves: Iterable[Any],
) -> Optional[Tuple[int, int]]:
    """Maze 规则直执行。

    若 subgoal_path 中出现 `(x,y) to (x2,y2)` 且 `(x,y)` 与当前玩家位置一致，
    则直接返回下一步目标 `(x2,y2)`；否则返回 None。
    """
    role = (player or "").strip().lower()
    state_obj = state
    if isinstance(state_obj, str):
        try:
            state_obj = json.loads(state_obj)
        except Exception:
            return None
    if not isinstance(state_obj, dict):
        return None

    s = state_obj.get(role) or {}
    try:
        cur = int(s.get("x")), int(s.get("y"))
    except Exception:
        return None

    for line in (subgoal_path or "").splitlines():
        m = _MAZE_STEP_RE.search(line)
        if not m:
            continue
        a = (int(m.group(1)), int(m.group(2)))
        b = (int(m.group(3)), int(m.group(4)))
        if a == cur:
            return b

    return None
