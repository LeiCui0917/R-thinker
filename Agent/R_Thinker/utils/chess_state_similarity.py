"""Chess 单节点语义相似度评分。

职责：
- 把当前 enhanced FEN 与单个 Chess 语义节点描述做匹配，输出 [0, 1] 分数。
- 只负责单节点打分；整棵树的遍历、检索与排序由 `tree_similarity.py` 负责。
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional

import chess


# ============================================================
# 模块一：基础结构
# ============================================================

MODE_EXACT = "exact"
MODE_ABS = "abs"
MODE_SET = "set"
MODE_ANY_NONZERO = "any_nonzero"
MODE_ANY_POS = "any_pos"
MODE_ANY_NEG = "any_neg"

ALIGN_FILE = "file"
ALIGN_RANK = "rank"
ALIGN_DIAGONAL = "diagonal"

_CACHED_COOLDOWN_DURATION: Optional[float] = None

_PIECE_NAME_TO_TYPE = {
	"pawn": chess.PAWN,
	"knight": chess.KNIGHT,
	"bishop": chess.BISHOP,
	"rook": chess.ROOK,
	"queen": chess.QUEEN,
	"king": chess.KING,
}


@dataclass(frozen=True)
class SemanticSpec:
	"""结构化的语义约束。"""

	# 参考国王颜色，以及要匹配的目标棋子颜色/类型。
	king_color: bool
	piece_color: bool
	piece_type: int
	# dx/dn 表示“棋子相对参考国王”的 file/rank 偏移约束。
	# *_mode 决定使用 exact / abs / set / any_* 哪种匹配规则。
	dx_mode: str
	dx: int
	dn_mode: str
	dn: int
	# None 表示文本未声明遮挡状态；True/False 表示显式要求。
	require_clear: Optional[bool]
	dx_set: Optional[List[int]] = None
	dn_set: Optional[List[int]] = None


@dataclass(frozen=True)
class EvalContext:
	"""Rule1 / Rule2 共用的评估上下文。"""

	board: chess.Board
	spec: SemanticSpec
	# king_sq / kf / kr 是参考国王的位置缓存，避免重复从 board 读取。
	king_sq: int
	kf: int
	kr: int
	# pcs_all: 所有同类目标棋子
	# pcs_ready: 当前不在 cooldown 中的棋子
	# pcs_selected: 本轮真正参与评分的棋子（优先 ready，否则 cooling）
	pcs_all: List[int]
	pcs_ready: List[int]
	pcs_selected: List[int]
	# alignment 是对滑子关系额外推断出的几何约束：同列 / 同行 / 对角线。
	alignment: Optional[str]


# ============================================================
# 模块二：FEN 与冷却时间
# ============================================================

def sanitize_fen(fen: str) -> str:
	"""把 enhanced FEN 规范化成 python-chess 可解析的标准 FEN。"""
	if not fen:
		return fen

	base = str(fen).split("|", 1)[0].strip()
	parts = base.split()
	if len(parts) < 6:
		return base
	if parts[1] not in ("w", "b"):
		parts[1] = "w"
	return " ".join(parts[:6])


def _get_configured_cooldown_duration() -> float:
	"""读取项目配置中的 chess 冷却时间。"""
	global _CACHED_COOLDOWN_DURATION
	if _CACHED_COOLDOWN_DURATION is not None:
		return float(_CACHED_COOLDOWN_DURATION)

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	config_path = os.path.join(base_dir, "config.json")
	with open(config_path, "r", encoding="utf-8") as f:
		cfg = json.load(f)
	val = float(cfg["game_settings"]["chess_cooldown"])
	_CACHED_COOLDOWN_DURATION = val
	return val


def _parse_cooldown_remaining_seconds(enhanced_fen_full: str) -> dict[str, float]:
	"""解析 enhanced FEN 里的 cooldown 字段为 {square: remaining_seconds}。"""
	if not enhanced_fen_full or "|cooldown:" not in str(enhanced_fen_full):
		return {}

	try:
		cooldown_part = str(enhanced_fen_full).split("|cooldown:", 1)[1]
		if not cooldown_part:
			return {}

		duration = _get_configured_cooldown_duration()
		now = time.time()
		out: dict[str, float] = {}
		for entry in cooldown_part.split(","):
			entry = entry.strip()
			if not entry or ":" not in entry:
				continue
			sq, ts_str = entry.split(":", 1)
			sq = sq.strip().lower()
			if len(sq) != 2:
				continue
			try:
				ts = float(ts_str)
			except Exception:
				continue
			remaining = (ts + duration) - now
			if remaining > 0:
				out[sq] = float(remaining)
		return out
	except Exception:
		return {}


def _cooldown_penalty_factor(remaining_seconds: float) -> float:
	"""把冷却剩余时间映射为 (0,1] 惩罚系数。"""
	min_factor = 0.6
	try:
		rem = float(remaining_seconds)
		if rem <= 0:
			return 1.0
		duration = float(_get_configured_cooldown_duration())
		if duration <= 0:
			return min_factor
		norm = max(0.0, min(1.0, rem / duration))
		return 1.0 - (1.0 - min_factor) * norm
	except Exception:
		return min_factor


def _split_pieces_by_cooldown(
	pieces: List[int],
	cooldown_remaining: dict[str, float],
) -> tuple[List[int], List[int]]:
	"""按冷却状态拆分棋子，返回 (ready, cooling)。"""
	ready: List[int] = []
	cooling: List[int] = []
	for psq in pieces:
		name = chess.square_name(psq).lower()
		rem = cooldown_remaining.get(name, 0.0)
		try:
			val = float(rem or 0.0)
		except Exception:
			val = 0.0
		if val > 0.0:
			cooling.append(psq)
		else:
			ready.append(psq)
	return ready, cooling


# ============================================================
# 模块三：棋盘几何工具
# ============================================================

def _count_blockers_line(board: chess.Board, a: int, b: int) -> int:
	"""统计同列或同行两格之间的挡子数量。"""
	af = chess.square_file(a)
	ar = chess.square_rank(a)
	bf = chess.square_file(b)
	br = chess.square_rank(b)

	if af == bf:
		step = 1 if br > ar else -1
		return sum(
			1
			for r in range(ar + step, br, step)
			if board.piece_at(chess.square(af, r))
		)

	if ar == br:
		step = 1 if bf > af else -1
		return sum(
			1
			for f in range(af + step, bf, step)
			if board.piece_at(chess.square(f, ar))
		)

	return 0


def _count_blockers_diag(board: chess.Board, a: int, b: int) -> int:
	"""统计同对角线两格之间的挡子数量。"""
	af = chess.square_file(a)
	ar = chess.square_rank(a)
	bf = chess.square_file(b)
	br = chess.square_rank(b)
	if abs(af - bf) != abs(ar - br):
		return 0

	df = 1 if bf > af else -1
	dr = 1 if br > ar else -1
	f, r = af + df, ar + dr
	cnt = 0
	while f != bf and r != br:
		if board.piece_at(chess.square(f, r)):
			cnt += 1
		f += df
		r += dr
	return cnt


# ============================================================
# 模块四：语义文本解析
# ============================================================

def _parse_offset_token(token: str) -> Optional[tuple[str, int]]:
	"""解析 `x` 或 `n` 后面的偏移 token。"""
	t = token.strip()
	if not t:
		return (MODE_EXACT, 0)

	if ("Î”" in t) or ("Î´" in t) or ("?" in t):
		if re.search(r"\-", t):
			return (MODE_ANY_NEG, 0)
		if re.search(r"\+", t):
			return (MODE_ANY_POS, 0)
		return (MODE_ANY_NONZERO, 0)

	m_pm = re.match(r"^(?:Â±|\+/-)\s*(\d+)$", t)
	if m_pm:
		return (MODE_ABS, int(m_pm.group(1)))

	m_signed = re.match(r"^([+\-])\s*(\d+)$", t)
	if m_signed:
		sgn = 1 if m_signed.group(1) == "+" else -1
		return (MODE_EXACT, sgn * int(m_signed.group(2)))

	m_plain = re.match(r"^(\d+)$", t)
	if m_plain:
		return (MODE_EXACT, int(m_plain.group(1)))
	return None


def _extract_delta_set(text: str, axis: str) -> Optional[List[int]]:
	"""提取形如 `Δx ∈ {-1,0,1}` 的显式取值集合。"""
	m_set = re.search(
		rf"(?:[Î”Î´?]{axis}|delta\s*{axis})\s*(?:âˆˆ|in)\s*\{{([^}}]+)\}}",
		text,
		flags=re.IGNORECASE,
	)
	if not m_set:
		return None

	nums = re.findall(r"[-+]?\d+", m_set.group(1))
	if not nums:
		return None

	vals: List[int] = []
	seen = set()
	for n in nums:
		try:
			v = int(n)
		except Exception:
			continue
		if v in seen:
			continue
		seen.add(v)
		vals.append(v)
	return vals


def _maybe_override_wildcard_to_set(
	parsed: tuple[str, int],
	token: str,
	allowed: Optional[List[int]],
) -> tuple[tuple[str, int], Optional[List[int]]]:
	"""若 token 是 `Δ` 通配形式，且文本给出显式集合，则改用 set 模式。"""
	if not allowed:
		return parsed, None
	mode, _ = parsed
	if not mode.startswith("any_"):
		return parsed, None
	mult = -1 if ("-" in token and "+" not in token) else 1
	return (MODE_SET, 0), [mult * int(v) for v in allowed]


def _parse_relative_subjects(low_text: str) -> Optional[tuple[bool, bool, int]]:
	"""提取棋子颜色、相对国王颜色与棋子类型。"""
	m_piece = re.search(r"\b(white|black)\s+(pawn|knight|bishop|rook|queen|king)\b", low_text)
	m_king = re.search(r"relative\s+to\s+(white|black)\s+king\b", low_text)
	if not m_piece or not m_king:
		return None

	piece_color = chess.WHITE if m_piece.group(1) == "white" else chess.BLACK
	king_color = chess.WHITE if m_king.group(1) == "white" else chess.BLACK
	piece_type = _PIECE_NAME_TO_TYPE.get(m_piece.group(2))
	if piece_type is None:
		return None
	return piece_color, king_color, piece_type


def _parse_relative_offsets(
	low_text: str,
	raw_text: str,
) -> Optional[tuple[tuple[str, int], tuple[str, int], Optional[List[int]], Optional[List[int]]]]:
	"""提取 `(x + a, n + b)` 形式的偏移约束与可选集合限制。"""
	# 这里只解析相对坐标主体，不负责棋子颜色、国王颜色和遮挡要求。
	m_xy = re.search(
		r"at\s*\(\s*x(?P<dx>[^,)]*)\s*,\s*n(?P<dn>[^)]*)\s*\)\s*relative\s+to\s*(?:(?:white|black)\s+king\s*)?\(\s*x\s*,\s*n\s*\)",
		low_text,
	)
	if not m_xy:
		return None

	dx_token = m_xy.group("dx")
	dn_token = m_xy.group("dn")
	dx_parsed = _parse_offset_token(dx_token)
	dn_parsed = _parse_offset_token(dn_token)
	if dx_parsed is None or dn_parsed is None:
		return None

	dx_set = _extract_delta_set(raw_text, "x")
	dn_set = _extract_delta_set(raw_text, "n")
	dx_parsed, dx_set = _maybe_override_wildcard_to_set(dx_parsed, dx_token, dx_set)
	dn_parsed, dn_set = _maybe_override_wildcard_to_set(dn_parsed, dn_token, dn_set)
	return dx_parsed, dn_parsed, dx_set, dn_set


def _parse_relative_relation(text: str) -> Optional[SemanticSpec]:
	"""解析 Chess 相对坐标语义，输出结构化 spec。"""
	if not text:
		return None
	low = str(text).lower()

	# 第一步：识别“谁相对谁”。
	subjects = _parse_relative_subjects(low)
	if subjects is None:
		return None
	piece_color, king_color, piece_type = subjects

	# 第二步：识别文本是否显式声明了路径遮挡要求。
	require_clear = None
	if re.search(r"\b(no\s+obstruction|without\s+obstruction|unobstructed)\b", low):
		require_clear = True
	elif re.search(r"\b(with\s+obstruction|obstructed|blocked|needs\s+clearing)\b", low):
		require_clear = False

	# 第三步：解析 `(x,n)` 相对偏移，以及可选的 Δ 取值集合。
	offsets = _parse_relative_offsets(low, str(text))
	if offsets is None:
		return None
	(dx_mode, dx), (dn_mode, dn), dx_set, dn_set = offsets

	return SemanticSpec(
		king_color=king_color,
		piece_color=piece_color,
		piece_type=piece_type,
		dx_mode=dx_mode,
		dx=dx,
		dn_mode=dn_mode,
		dn=dn,
		require_clear=require_clear,
		dx_set=dx_set,
		dn_set=dn_set,
	)


# ============================================================
# 模块五：公共评估上下文
# ============================================================

def _is_sliding_piece(piece_type: int) -> bool:
	"""是否为依赖路径遮挡的滑子：车、象、后。"""
	return int(piece_type) in {chess.ROOK, chess.BISHOP, chess.QUEEN}


def _target_alignment_from_spec(spec: SemanticSpec) -> Optional[str]:
	"""由目标 `(dx, dn)` 推断其可能落在哪类射线上。"""
	if spec.dx_mode == MODE_ABS or spec.dn_mode == MODE_ABS:
		return None

	if spec.dx_mode == MODE_EXACT and spec.dn_mode == MODE_EXACT:
		if spec.dx == 0 and spec.dn != 0:
			return ALIGN_FILE
		if spec.dn == 0 and spec.dx != 0:
			return ALIGN_RANK
		if spec.dx != 0 and spec.dn != 0 and abs(spec.dx) == abs(spec.dn):
			return ALIGN_DIAGONAL
		return None

	if spec.dn_mode == MODE_EXACT and spec.dn == 0 and spec.dx_mode in {MODE_ANY_NONZERO, MODE_ANY_POS, MODE_ANY_NEG}:
		return ALIGN_RANK
	if spec.dx_mode == MODE_EXACT and spec.dx == 0 and spec.dn_mode in {MODE_ANY_NONZERO, MODE_ANY_POS, MODE_ANY_NEG}:
		return ALIGN_FILE
	if spec.dx_mode in {MODE_ANY_NONZERO, MODE_ANY_POS, MODE_ANY_NEG} and spec.dn_mode in {MODE_ANY_NONZERO, MODE_ANY_POS, MODE_ANY_NEG}:
		if spec.piece_type in {chess.BISHOP, chess.QUEEN}:
			return ALIGN_DIAGONAL
	return None


def _rule2_applicable(spec: SemanticSpec, alignment: Optional[str]) -> bool:
	"""集中判断 Rule2 是否适用。"""
	return spec.require_clear is not None and _is_sliding_piece(spec.piece_type) and alignment is not None


def _build_eval_context(
	board: chess.Board,
	spec: SemanticSpec,
	cooldown_remaining: dict[str, float],
) -> Optional[EvalContext]:
	"""构造 Rule1 / Rule2 共用的评估上下文。"""
	king_sq = board.king(spec.king_color)
	if king_sq is None:
		return None

	pcs_all = list(board.pieces(spec.piece_type, spec.piece_color))
	if not pcs_all:
		return None

	# 项目约定：若存在 ready 棋子，则忽略 cooling 棋子；只有全都在冷却中时才退回使用 cooling。
	pcs_ready, pcs_cooling = _split_pieces_by_cooldown(pcs_all, cooldown_remaining)
	pcs_selected = pcs_ready if pcs_ready else pcs_cooling
	if not pcs_selected:
		return None

	return EvalContext(
		board=board,
		spec=spec,
		king_sq=king_sq,
		kf=chess.square_file(king_sq),
		kr=chess.square_rank(king_sq),
		pcs_all=pcs_all,
		pcs_ready=pcs_ready,
		pcs_selected=pcs_selected,
		alignment=_target_alignment_from_spec(spec) if _is_sliding_piece(spec.piece_type) else None,
	)


# ============================================================
# 模块六：Rule1 位置相似度
# ============================================================

def _offset_matches(delta: int, mode: str, target: int) -> bool:
	"""判断某个偏移是否满足 exact / abs / any_* 约束。"""
	if mode == MODE_ANY_NONZERO:
		return int(delta) != 0
	if mode == MODE_ANY_POS:
		return int(delta) > 0
	if mode == MODE_ANY_NEG:
		return int(delta) < 0
	if mode == MODE_ABS:
		return abs(delta) == abs(int(target))
	return int(delta) == int(target)


def _alignment_holds(alignment: str, dx: int, dn: int) -> bool:
	"""检查 `(dx, dn)` 是否位于指定对齐类型上。"""
	dx = int(dx)
	dn = int(dn)
	if alignment == ALIGN_FILE:
		return dx == 0 and dn != 0
	if alignment == ALIGN_RANK:
		return dn == 0 and dx != 0
	if alignment == ALIGN_DIAGONAL:
		return dx != 0 and dn != 0 and abs(dx) == abs(dn)
	return False


def _relation_holds(ctx: EvalContext) -> bool:
	"""判断当前棋盘是否存在满足目标 `(dx, dn)` 的棋子。"""
	dx_set_s = set(ctx.spec.dx_set or []) if ctx.spec.dx_mode == MODE_SET else None
	dn_set_s = set(ctx.spec.dn_set or []) if ctx.spec.dn_mode == MODE_SET else None

	for psq in ctx.pcs_selected:
		pf = chess.square_file(psq)
		pr = chess.square_rank(psq)
		dx = int(pf - ctx.kf)
		dn = int(pr - ctx.kr)

		# 对滑子关系施加几何对齐限制，避免 Bishop/Queen 的 Δ 语义被匹配到明显错误的位置。
		if ctx.alignment is not None and not _alignment_holds(ctx.alignment, dx, dn):
			continue

		if ctx.spec.dx_mode == MODE_SET and dx_set_s is not None:
			dx_ok = dx in dx_set_s
		else:
			dx_ok = _offset_matches(dx, ctx.spec.dx_mode, ctx.spec.dx)

		if ctx.spec.dn_mode == MODE_SET and dn_set_s is not None:
			dn_ok = dn in dn_set_s
		else:
			dn_ok = _offset_matches(dn, ctx.spec.dn_mode, ctx.spec.dn)

		if dx_ok and dn_ok:
			return True
	return False


def _min_cooldown_remaining_seconds(
	pcs: List[int],
	cooldown_remaining: dict[str, float],
) -> float:
	"""返回候选棋子中的最小剩余冷却时间。"""
	best = None
	for psq in pcs:
		rem = cooldown_remaining.get(chess.square_name(psq).lower(), 0.0)
		try:
			val = float(rem or 0.0)
		except Exception:
			val = 0.0
		best = val if best is None else min(best, val)
	return float(best or 0.0)


def _wildcard_cost(delta: int, mode: str, target: int, allowed_set: Optional[List[int]] = None) -> int:
	"""计算到最近满足约束值的距离代价。"""
	if mode == MODE_SET:
		if not allowed_set:
			return 999
		if int(delta) in set(int(v) for v in allowed_set):
			return 0
		return min(abs(int(delta) - int(v)) for v in allowed_set)
	if mode == MODE_ANY_NONZERO:
		return 0 if delta != 0 else 1
	if mode == MODE_ANY_POS:
		return 0 if delta > 0 else int(1 - delta)
	if mode == MODE_ANY_NEG:
		return 0 if delta < 0 else int(delta + 1)
	if mode == MODE_ABS:
		return abs(abs(delta) - abs(int(target)))
	return abs(delta - int(target))


def _position_similarity_rule1(
	ctx: EvalContext,
	cooldown_remaining: dict[str, float],
) -> float:
	"""Rule1：根据相对位置关系计算位置相似度。"""
	# 若当前棋盘上已经存在完全满足关系的棋子，直接给高分。
	if _relation_holds(ctx):
		base = 1.0
		if not ctx.pcs_ready:
			# 若只能依赖 cooling 棋子，则再乘一个冷却惩罚。
			base *= _cooldown_penalty_factor(
				_min_cooldown_remaining_seconds(ctx.pcs_selected, cooldown_remaining)
			)
		return float(base)

	# 否则退化为“距离最近的候选关系有多近”的近似评分。
	best_score_ready = 0.0
	best_score_cooling = 0.0
	for psq in ctx.pcs_all:
		name = chess.square_name(psq).lower()
		rem = float(cooldown_remaining.get(name, 0.0) or 0.0)
		in_cd = rem > 0.0
		if ctx.pcs_ready and in_cd:
			continue

		pf = chess.square_file(psq)
		pr = chess.square_rank(psq)
		dx = int(pf - ctx.kf)
		dn = int(pr - ctx.kr)

		dx_cost = _wildcard_cost(dx, ctx.spec.dx_mode, ctx.spec.dx, ctx.spec.dx_set)
		dn_cost = _wildcard_cost(dn, ctx.spec.dn_mode, ctx.spec.dn, ctx.spec.dn_set)

		# 对滑子额外加入对齐代价，避免仅靠 dx/dn 通配造成语义过宽。
		align_cost = 0
		if ctx.alignment is not None:
			if ctx.alignment == ALIGN_FILE:
				align_cost = abs(dx)
			elif ctx.alignment == ALIGN_RANK:
				align_cost = abs(dn)
			elif ctx.alignment == ALIGN_DIAGONAL:
				align_cost = abs(abs(dx) - abs(dn))

		cost = int(max(0, dx_cost) + max(0, dn_cost) + max(0, align_cost))
		base = 1.0 / (1.0 + float(cost))
		if in_cd:
			base *= _cooldown_penalty_factor(rem)
			best_score_cooling = max(best_score_cooling, base)
		else:
			best_score_ready = max(best_score_ready, base)

	return float(best_score_ready if best_score_ready > 0.0 else best_score_cooling)


# ============================================================
# 模块七：Rule2 路径遮挡相似度
# ============================================================

def _path_similarity_rule2(
	ctx: EvalContext,
	cooldown_remaining: dict[str, float],
) -> float:
	"""Rule2：根据路径是否遮挡计算相似度。"""
	if not _rule2_applicable(ctx.spec, ctx.alignment):
		return 0.0

	# Rule2 只在已经能推断出明确射线关系时才有意义。
	best = 0.0
	for psq in ctx.pcs_selected:
		rem = float(cooldown_remaining.get(chess.square_name(psq).lower(), 0.0) or 0.0)
		pf = chess.square_file(psq)
		pr = chess.square_rank(psq)

		aligned = False
		blockers = 0
		if ctx.alignment == ALIGN_FILE and pf == ctx.kf:
			aligned = True
			blockers = _count_blockers_line(ctx.board, psq, ctx.king_sq)
		elif ctx.alignment == ALIGN_RANK and pr == ctx.kr:
			aligned = True
			blockers = _count_blockers_line(ctx.board, psq, ctx.king_sq)
		elif ctx.alignment == ALIGN_DIAGONAL and abs(pf - ctx.kf) == abs(pr - ctx.kr):
			aligned = True
			blockers = _count_blockers_diag(ctx.board, psq, ctx.king_sq)

		if not aligned:
			continue

		if bool(ctx.spec.require_clear):
			# 要求通路畅通：挡子越少越好，0 个挡子最优。
			sim = 1.0 / (1.0 + float(max(0, blockers)))
		else:
			# 要求被阻挡：至少要有 1 个挡子，且越接近 1 个越好。
			sim = 0.0 if blockers <= 0 else 1.0 / (1.0 + float(max(0, blockers - 1)))

		if not ctx.pcs_ready and rem > 0:
			sim *= _cooldown_penalty_factor(rem)
		best = max(best, sim)
	return float(best)


# ============================================================
# 模块八：总入口
# ============================================================

def _relation_score_v2(
	ctx: EvalContext,
	*,
	cooldown_remaining: Optional[dict[str, float]] = None,
	w_pos: float = 0.5,
	w_path: float = 0.5,
) -> float:
	"""融合 Rule1 / Rule2 得到最终相似度。"""
	cd = cooldown_remaining or {}
	s1 = _position_similarity_rule1(ctx, cd)
	# 若 Rule2 不适用，就退回到纯位置相似度。
	if not _rule2_applicable(ctx.spec, ctx.alignment):
		return float(s1)
	s2 = _path_similarity_rule2(ctx, cd)
	return float((w_pos * s1) + (w_path * s2))


def _has_any_onboard_target_square(board: chess.Board, spec: SemanticSpec) -> bool:
	"""检查相对偏移约束是否至少对应一个棋盘内格子。"""
	king_sq = board.king(spec.king_color)
	if king_sq is None:
		return False

	kf = chess.square_file(king_sq)
	kr = chess.square_rank(king_sq)

	def _in_bounds(f: int, r: int) -> bool:
		return 0 <= int(f) <= 7 and 0 <= int(r) <= 7

	def _axis_candidates(mode: str, val: int, allowed: Optional[List[int]]) -> List[int]:
		if mode == MODE_EXACT:
			return [int(val)]
		if mode == MODE_ABS:
			v = abs(int(val))
			return [v, -v]
		if mode == MODE_SET:
			return [int(x) for x in (allowed or [])]
		if mode == MODE_ANY_POS:
			return [1]
		if mode == MODE_ANY_NEG:
			return [-1]
		if mode == MODE_ANY_NONZERO:
			return [1, -1]
		return []

	dx_cands = _axis_candidates(spec.dx_mode, spec.dx, spec.dx_set)
	dn_cands = _axis_candidates(spec.dn_mode, spec.dn, spec.dn_set)
	if not dx_cands or not dn_cands:
		return False

	for ddx in dx_cands:
		for ddn in dn_cands:
			if _in_bounds(kf + int(ddx), kr + int(ddn)):
				return True
	return False


def chess_semantic_similarity_score(
	fen: str,
	semantic_text: str,
) -> float:
	"""对外入口：计算 Chess 单节点语义相似度。"""
	if not fen or not semantic_text:
		return 0.0

	try:
		board = chess.Board(sanitize_fen(fen))
	except Exception:
		return 0.0

	cooldown_remaining = _parse_cooldown_remaining_seconds(fen)
	spec = _parse_relative_relation(str(semantic_text))
	if spec is None:
		return 0.0
	# 对明显越界、根本不可能落到棋盘内的偏移关系做硬拒绝。
	if not _has_any_onboard_target_square(board, spec):
		return 0.0

	ctx = _build_eval_context(board, spec, cooldown_remaining)
	if ctx is None:
		return 0.0
	return _relation_score_v2(ctx, cooldown_remaining=cooldown_remaining)
