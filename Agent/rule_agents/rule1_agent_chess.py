import chess
import math
import time
from typing import Dict, Tuple
from Env.ChessEnv.RealtimeChessEnv import RealtimeChessEnv  # Import environment class


class RuleAgent:
    """
    Stockfish-style hybrid rule agent for realtime chess:
    - Iterative deepening alpha-beta (negamax) with move ordering
    - Lightweight NNUE-style static evaluation (feature accumulator + linear head)
    - Realtime-safe time budgeting per decision
    """

    # Centipawn-like piece values.
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }

    # Simple piece-square priors (white perspective); black uses mirrored index.
    PST_PAWN = [
        0, 0, 0, 0, 0, 0, 0, 0,
        8, 8, 8, 10, 10, 8, 8, 8,
        5, 6, 7, 9, 9, 7, 6, 5,
        3, 4, 5, 8, 8, 5, 4, 3,
        2, 3, 4, 7, 7, 4, 3, 2,
        1, 2, 3, 5, 5, 3, 2, 1,
        0, 1, 2, -2, -2, 2, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ]
    PST_KNIGHT = [
        -20, -10, -8, -8, -8, -8, -10, -20,
        -10, -2, 0, 0, 0, 0, -2, -10,
        -8, 0, 6, 8, 8, 6, 0, -8,
        -8, 2, 8, 10, 10, 8, 2, -8,
        -8, 0, 8, 10, 10, 8, 0, -8,
        -8, 2, 6, 8, 8, 6, 2, -8,
        -10, -2, 0, 2, 2, 0, -2, -10,
        -20, -10, -8, -8, -8, -8, -10, -20,
    ]
    PST_BISHOP = [
        -10, -8, -6, -6, -6, -6, -8, -10,
        -8, 0, 2, 2, 2, 2, 0, -8,
        -6, 2, 5, 6, 6, 5, 2, -6,
        -6, 2, 6, 8, 8, 6, 2, -6,
        -6, 2, 6, 8, 8, 6, 2, -6,
        -6, 2, 5, 6, 6, 5, 2, -6,
        -8, 0, 2, 2, 2, 2, 0, -8,
        -10, -8, -6, -6, -6, -6, -8, -10,
    ]
    PST_ROOK = [
        0, 0, 2, 4, 4, 2, 0, 0,
        -2, 0, 2, 4, 4, 2, 0, -2,
        -2, 0, 2, 4, 4, 2, 0, -2,
        -2, 0, 2, 4, 4, 2, 0, -2,
        -2, 0, 2, 4, 4, 2, 0, -2,
        -2, 0, 2, 4, 4, 2, 0, -2,
        2, 4, 6, 8, 8, 6, 4, 2,
        0, 0, 2, 4, 4, 2, 0, 0,
    ]
    PST_QUEEN = [
        -8, -6, -4, -2, -2, -4, -6, -8,
        -6, -2, 0, 0, 0, 0, -2, -6,
        -4, 0, 2, 2, 2, 2, 0, -4,
        -2, 0, 2, 4, 4, 2, 0, -2,
        -2, 0, 2, 4, 4, 2, 0, -2,
        -4, 0, 2, 2, 2, 2, 0, -4,
        -6, -2, 0, 0, 0, 0, -2, -6,
        -8, -6, -4, -2, -2, -4, -6, -8,
    ]
    PST_KING = [
        -20, -24, -24, -28, -28, -24, -24, -20,
        -18, -22, -22, -26, -26, -22, -22, -18,
        -16, -20, -20, -24, -24, -20, -20, -16,
        -12, -16, -16, -20, -20, -16, -16, -12,
        -8, -12, -12, -16, -16, -12, -12, -8,
        -6, -10, -10, -14, -14, -10, -10, -6,
        8, 4, 0, -4, -4, 0, 4, 8,
        12, 16, 8, 0, 0, 8, 16, 12,
    ]
    PST_BY_PIECE = {
        chess.PAWN: PST_PAWN,
        chess.KNIGHT: PST_KNIGHT,
        chess.BISHOP: PST_BISHOP,
        chess.ROOK: PST_ROOK,
        chess.QUEEN: PST_QUEEN,
        chess.KING: PST_KING,
    }

    def __init__(self, env: RealtimeChessEnv, delay_s: float = 0.0):
        """Initialize Agent, receive real-time chess environment instance"""
        self.env = env  # Store environment instance for calling parsing methods
        self.delay_s = max(0.0, float(delay_s or 0.0))
        self.last_action_source = "rule"
        self.max_depth = 4
        self.base_time_budget_s = 0.12
        self.tt: Dict[Tuple[str, int], float] = {}
        self._deadline = 0.0

    def _timed_out(self) -> bool:
        return time.time() >= self._deadline

    def _parse_board(self, enhanced_fen_full: str, color: str) -> chess.Board:
        enhanced_fen = self.env.enhancedFENfull_to_enhancedFEN(enhanced_fen_full)
        standard_fen = self.env.enhancedFEN_to_standard(enhanced_fen, color)
        return chess.Board(standard_fen)

    def _pst_value(self, piece_type: int, square: int, piece_color: bool) -> int:
        table = self.PST_BY_PIECE[piece_type]
        idx = square if piece_color == chess.WHITE else chess.square_mirror(square)
        return int(table[idx])

    def _nnue_like_eval(self, board: chess.Board, perspective: bool) -> float:
        # Terminal handling for realtime variant where king capture can happen.
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king is None and black_king is None:
            return 0.0
        if white_king is None:
            return -1e6 if perspective == chess.WHITE else 1e6
        if black_king is None:
            return 1e6 if perspective == chess.WHITE else -1e6

        # NNUE-style: feature accumulation then linear combination.
        white_mat = 0
        black_mat = 0
        white_pos = 0
        black_pos = 0

        for sq, piece in board.piece_map().items():
            val = self.PIECE_VALUES[piece.piece_type]
            pst = self._pst_value(piece.piece_type, sq, piece.color)
            if piece.color == chess.WHITE:
                white_mat += val
                white_pos += pst
            else:
                black_mat += val
                black_pos += pst

        turn_before = board.turn
        board.turn = chess.WHITE
        white_mob = sum(1 for _ in board.pseudo_legal_moves)
        board.turn = chess.BLACK
        black_mob = sum(1 for _ in board.pseudo_legal_moves)
        board.turn = turn_before

        white_score = white_mat + white_pos + 3 * white_mob
        black_score = black_mat + black_pos + 3 * black_mob
        score = float(white_score - black_score)
        return score if perspective == chess.WHITE else -score

    def _move_order_key(self, board: chess.Board, move: chess.Move) -> tuple:
        is_capture = board.is_capture(move)
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        victim_v = self.PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
        attacker_v = self.PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
        mvv_lva = victim_v - attacker_v
        is_promo = 1 if move.promotion else 0
        return (1 if is_capture else 0, is_promo, mvv_lva)

    def _is_king_capture(self, board: chess.Board, move: chess.Move, perspective: bool) -> bool:
        if not board.is_capture(move):
            return False
        victim = board.piece_at(move.to_square)
        if victim is None:
            return False
        return victim.piece_type == chess.KING and victim.color != perspective

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, perspective: bool) -> float:
        if self._timed_out():
            raise TimeoutError

        if depth <= 0:
            return self._nnue_like_eval(board, perspective)

        key = (board.fen(en_passant='fen'), depth)
        if key in self.tt:
            return self.tt[key]

        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            val = self._nnue_like_eval(board, perspective)
            self.tt[key] = val
            return val

        moves = list(board.pseudo_legal_moves)
        if not moves:
            val = self._nnue_like_eval(board, perspective)
            self.tt[key] = val
            return val

        moves.sort(key=lambda m: self._move_order_key(board, m), reverse=True)
        best = -math.inf
        for mv in moves:
            board.push(mv)
            score = -self._negamax(board, depth - 1, -beta, -alpha, perspective)
            board.pop()

            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        self.tt[key] = best
        return best
    
    def get_action(self, enhanced_FEN_full, color, legal_moves):
        self.last_action_source = "rule"
        parsed_moves = [chess.Move.from_uci(uci_move) for uci_move in (legal_moves or [])]
        if not parsed_moves:
            return None

        board = self._parse_board(enhanced_FEN_full, color)
        perspective = chess.WHITE if color == 'w' else chess.BLACK

        # Root search only considers realtime-filtered legal moves.
        root_moves = [m for m in parsed_moves if m in board.pseudo_legal_moves]
        if not root_moves:
            root_moves = parsed_moves

        # Realtime rule priority: capturing opponent king ends the game immediately.
        for mv in root_moves:
            if self._is_king_capture(board, mv, perspective):
                selected = mv.uci()
                if self.delay_s > 0.0:
                    time.sleep(self.delay_s)
                return selected

        budget = min(0.22, self.base_time_budget_s + 0.003 * len(root_moves))
        self._deadline = time.time() + max(0.03, budget)

        best_move = root_moves[0]
        best_score = -math.inf

        try:
            for depth in range(1, self.max_depth + 1):
                local_best_move = best_move
                local_best_score = -math.inf

                ordered = sorted(root_moves, key=lambda m: self._move_order_key(board, m), reverse=True)
                for mv in ordered:
                    if self._timed_out():
                        raise TimeoutError
                    board.push(mv)
                    score = -self._negamax(board, depth - 1, -math.inf, math.inf, perspective)
                    board.pop()

                    if score > local_best_score:
                        local_best_score = score
                        local_best_move = mv

                best_move = local_best_move
                best_score = local_best_score
                _ = best_score
        except TimeoutError:
            pass

        selected = best_move.uci()
        if self.delay_s > 0.0:
            time.sleep(self.delay_s)
        return selected
