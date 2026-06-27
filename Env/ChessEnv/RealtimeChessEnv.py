"""
Real-time Chess Rules Description

1. Asynchronous Moves: Both sides can move at any time, no longer strictly taking turns.
2. Cooldown Mechanism: After each piece moves, it enters a cooldown period and cannot move again during this time.
3. Move Validation: Only pieces with 0 cooldown can be moved; illegal moves will be rejected.
4. Win/Loss Judgment: Win/loss rules are consistent with standard chess (checkmate, stalemate, etc.).
5. Thread Safety: All operations are locked to support multi-agent/multi-user concurrency.
6. Extensibility: Cooldown time is configurable, supporting different gameplay and experiments.

Enhanced FEN Format Description:
Standard FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Enhanced FEN: Standard FEN with current turn replaced by * + |cooldown:square:timestamp,square:timestamp

Example:
standard_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
enhanced_FEN_base = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR * KQkq - 0 1"
enhanced_FEN_full = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR * KQkq - 0 1|cooldown:e2:1712345678.123"

Format Details:
- Separator: |cooldown: Identifies the start of cooldown information
- Cooldown Entry: square:timestamp (multiple entries separated by commas)
- Square: Chess coordinates (e.g., e2, e4)
- Start Timestamp: Unix timestamp, precise to milliseconds

Real-time Chess Environment Class
"""

import chess
import time
import threading
import json
import os
from typing import Dict, Any, List

class RealtimeChessEnv:
    """
    Real-time Chess Environment (Revised)
    Supports cooldown mechanism, legal move validation, enhanced FEN state, and win/loss judgment.
    """

    def __init__(self, cooldown: float | None = None):
        # Initialize board and parameters
        self.board = chess.Board()

        if cooldown is None:
            base_dir = os.path.dirname(__file__)
            candidate_paths = [
                os.path.abspath(os.path.join(base_dir, '..', '..', 'config.json')),  # project root
                os.path.abspath(os.path.join(base_dir, '..', 'config.json')),        # legacy fallback
            ]
            config_path = next((p for p in candidate_paths if os.path.exists(p)), None)
            if config_path is None:
                raise FileNotFoundError(
                    f"config.json not found. Tried: {candidate_paths}"
                )
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            gs = cfg['game_settings']
            cooldown = float(gs['chess_cooldown'])

        self.cooldown = float(cooldown)       # Cooldown time per move (seconds)
        self.lock = threading.RLock()
        
        # Step counters
        self.white_steps = 0
        self.black_steps = 0

        # New architecture fields
        self.enhanced_FEN_base = self.standardFEN_to_enhanced(self.board.fen())
        # Cooldown state: square -> start_time
        self.cooldown_states: Dict[str, float] = {}  

        # Pause control (cooldown does not elapse during pause)
        self._paused = False
        self._pause_started = None
        self._pause_total = 0.0

    # =========================================================
    # 1. FEN Extraction and Combination
    # =========================================================
    def standardFEN_to_enhanced(self,fen: str) -> str:
        """Change turn indicator ('w'/'b') in standard FEN to '*'"""
        parts = fen.split(' ')
        if len(parts) > 1:
            parts[1] = '*'
        return ' '.join(parts)

    def enhancedFEN_to_standard(self, fen: str, color: str) -> str:
        """Change '*' in enhanced FEN back to specified color ('w'/'b')"""
        parts = fen.split(' ')
        if len(parts) > 1 and parts[1] == '*':
            parts[1] = color
        return ' '.join(parts)
    
    def enhancedFENfull_to_enhancedFEN(self, fen: str) -> str:
        # Parse enhanced FEN (separate standard FEN and cooldown info)
        if '|cooldown:' in fen:
            enhancedFEN = fen.split('|cooldown:', 1)[0]
        else:
            enhancedFEN = fen
        return enhancedFEN  # Fix: Return parsed base FEN, not instance variable
    
    def enhancedFENfull_to_cooldown_part(self, fen: str) -> Dict[str, float]:
        """Parse cooldown info in FEN into {square: timestamp} dictionary"""
        if '|cooldown:' not in fen:
            return {}  # Return empty dict if no cooldown info
        
        # Extract cooldown part (e.g., "e2:1712345678.123,a7:1712345679.456")
        cooldown_part = fen.split('|cooldown:', 1)[1]
        cooldown_dict = {}
        
        # Parse each cooldown entry
        for entry in cooldown_part.split(','):
            if not entry.strip():
                continue  # Skip empty entries
            # Split square and timestamp (e.g., "e2:1712345678.123" -> ("e2", "1712345678.123"))
            square, timestamp_str = entry.split(':', 1)
            timestamp = float(timestamp_str)  # Convert to float timestamp
            cooldown_dict[square.strip()] = timestamp
        
        return cooldown_dict  # Return parsed cooldown dict

    def get_enhancedFENfull(self) -> str:
        """
        Return Enhanced FEN:
        Base layout + |cooldown:e2:172345.123,e7:172346.222
        """
        with self.lock:
            self.cleanup_expired_cooldowns()
            if not self.cooldown_states:
                return self.enhanced_FEN_base
            # Output timestamps according to "real timeline": stored as real start_ts, add accumulated pause time when outputting,
            # so that frontend can align with server's effective time when calculating with real now.
            # Since _now_effective = time.time() - paused_total, we have paused_total = time.time() - _now_effective()
            paused_total = time.time() - self._now_effective()
            cooldown_entries = [
                f"{sq}:{(ts + paused_total):.3f}" for sq, ts in self.cooldown_states.items()
            ]
            return self.enhanced_FEN_base + "|cooldown:" + ",".join(cooldown_entries)

    # =========================================================
    # 2. Cooldown Mechanism
    # =========================================================
    def _add_cooldown(self, square: str):
         """Add cooldown for specified square (e.g., 'e2')"""
         with self.lock:
             # Store as start of "effective time" (real_now - paused_total_at_start),
             # so output only needs to add current accumulated pause to get correct real reference time, avoiding over-compensation from multiple pauses.
             self.cooldown_states[square] = self._now_effective()

    def _is_piece_in_cooldown(self, square: str) -> bool:
         """Check if the piece on the square is in cooldown"""
         with self.lock:
             self.cleanup_expired_cooldowns()
             ts = self.cooldown_states.get(square)
             if ts is None:
                 return False
             # Use effective time (excluding pause duration)
             return (self._now_effective() - ts) < self.cooldown

    def cleanup_expired_cooldowns(self):
        """Clear expired cooldown squares"""
        with self.lock:
            # Use effective time (excluding pause duration)
            now = self._now_effective()
            expired = [
                sq for sq, ts in self.cooldown_states.items()
                if now - ts >= self.cooldown
            ]
            for sq in expired:
                del self.cooldown_states[sq]
    # ---- Pause/Resume and Time Basis ----
    def _now_effective(self) -> float:
        """Effective time = Current real time - Accumulated pause duration."""
        t = time.time()
        paused_total = self._pause_total + (max(0.0, t - self._pause_started) if self._paused and self._pause_started is not None else 0.0)
        return t - paused_total

    def pause(self):
        with self.lock:
            if not self._paused:
                self._paused = True
                self._pause_started = time.time()

    def resume(self):
        with self.lock:
            if self._paused:
                if self._pause_started is not None:
                    self._pause_total += max(0.0, time.time() - self._pause_started)
                self._paused = False
                self._pause_started = None

    # =========================================================
    # 3. Move Legality
    # =========================================================
    def _is_move_legal_realtime(self, move_uci: str, color: str) -> bool:
        """Real-time legality check (considering cooldown)"""
        with self.lock:
            # 1. Convert enhanced FEN to standard FEN for specified color
            standard_fen = self.enhancedFEN_to_standard(self.enhanced_FEN_base, color)
            # 2. Create temporary board
            temp_board = chess.Board(standard_fen)
            move = chess.Move.from_uci(move_uci)
            
            # 3. Legality check (Pseudo-legal moves only, ignoring king safety)
            if move not in temp_board.pseudo_legal_moves:
                return False
                
            # 4. Cooldown check
            square_name = chess.square_name(move.from_square)
            if self._is_piece_in_cooldown(square_name):
                return False
            return True

    def get_legal_moves_for_color(self, color: str) -> List[str]:
        """Get all legal moves for current color (Real-time version)"""
        with self.lock:
            # 1. Convert enhanced FEN to standard FEN for specified color
            standard_fen = self.enhancedFEN_to_standard(self.enhanced_FEN_base, color)
            # 2. Create temporary board
            temp_board = chess.Board(standard_fen)
            moves = []
            # Use pseudo_legal_moves to allow moves that might expose king to check
            for move in temp_board.pseudo_legal_moves:
                square_name = chess.square_name(move.from_square)
                if not self._is_piece_in_cooldown(square_name):
                    moves.append(move.uci())
            return moves

    def process_move_action(self, move_uci: str, color: str) -> bool:
        """Submit move action"""
        with self.lock:
           
            # 1. Get enhanced state FEN and convert to standard FEN (restore turn)
            standard_fen = self.enhancedFEN_to_standard(self.enhanced_FEN_base, color)

            # 2. Create temporary board using standard FEN
            temp_board = chess.Board(standard_fen)
            move = chess.Move.from_uci(move_uci)

            # 3. Check if move is legal (using pseudo-legal moves + cooldown)
            if move not in temp_board.pseudo_legal_moves:
                print(f"❌ Illegal Move: {move_uci}")
                print(f"Current Board: {temp_board.fen()}")
                return False
            if self._is_piece_in_cooldown(chess.square_name(move.from_square)):
                print(f"⏳ Move during Cooldown: {move_uci}")
                return False

            # 4. Execute move (update board)
            temp_board.push(move)

            # 5. Sync enhanced FEN (reset turn to *)
            self.enhanced_FEN_base = self.standardFEN_to_enhanced(temp_board.fen())

            # 6. Update cooldown state
            self._add_cooldown(chess.square_name(move.to_square))
            
            # 7. Update step count
            if color == 'w':
                self.white_steps += 1
            else:
                self.black_steps += 1
                
            return True
        
    # =========================================================
    # 5. Win/Loss and Status Judgment
    # =========================================================

    def check_game_status(self) -> Dict[str, Any]:
        """Check overall game status (use snapshot to avoid holding lock for too long)"""
        # Get FEN snapshot only once to avoid holding lock during calculation
        with self.lock:
            fen_snapshot = self.enhanced_FEN_base
            w_steps = self.white_steps
            b_steps = self.black_steps
            
        return {
            "white": self._check_color_status(fen_snapshot, "w"),
            "black": self._check_color_status(fen_snapshot, "b"),
            "white_steps": w_steps,
            "black_steps": b_steps
        }

    def _check_color_status(self, enhanced_fen_base_temp: str, color: str) -> Dict[str, Any]:
        """Determine status of specified color"""
        # Build temporary board using passed FEN snapshot to avoid holding lock in this function
        temp_board = chess.Board(self.enhancedFEN_to_standard(enhanced_fen_base_temp, color))

        king_missing = temp_board.king(chess.WHITE if color == 'w' else chess.BLACK) is None
        
        return {
            "king_missing": king_missing,
            "checkmate": False,
            "insufficient_material": False,
            "repetition": False,
            "fifty_moves": False,
            "game_over": king_missing,
        }

    def get_termination_reason(self, game_status: Dict[str, Any] = None) -> str:
        """
        Analyze given game_status and return human-readable termination reason string.
        """
        white_status = game_status.get("white", {})
        black_status = game_status.get("black", {})

        # If neither side has reached termination condition, return In Progress to avoid misleading frontend text
        if not (white_status.get("game_over") or black_status.get("game_over")):
            return "In Progress"

        # Priority: King Missing

        # King Missing
        if white_status.get("king_missing") and not black_status.get("king_missing"):
            return "White King captured, Black wins"
        if black_status.get("king_missing") and not white_status.get("king_missing"):
            return "Black King captured, White wins"
        if white_status.get("king_missing") and black_status.get("king_missing"):
            return "Both Kings missing - Illegal state, Draw"
        
        return "Game Over (Reason Unknown)"

    # =========================================================
    # VI. Reset Game
    # =========================================================

    def reset(self):
        """Reset Game"""
        with self.lock:
            self.board.reset()
            self.cooldown_states.clear()
            self.enhanced_FEN_base = self.standardFEN_to_enhanced(self.board.fen())
            # Synchronously reset pause-related states
            self._paused = False
            self._pause_started = None
            self._pause_total = 0.0