"""Agent/utils/fen_translator.py

【模块：翻译器（Chess）】

职责：
- 将 enhanced FEN（可能带 `|cooldown:` 扩展信息）翻译成自然语言描述，供提示词使用。

说明：
- 本文件偏“展示/提示词辅助”，不参与决策逻辑。
- 依赖 `python-chess` 来解析棋盘。
"""

import chess
import json
import os
import time

class FenTranslator:
    """把 enhanced FEN 翻译成自然语言棋盘描述。"""
    def __init__(self):
        self.piece_names_en = {
            'P': 'White Pawn', 'N': 'White Knight', 'B': 'White Bishop', 'R': 'White Rook', 'Q': 'White Queen', 'K': 'White King',
            'p': 'Black Pawn', 'n': 'Black Knight', 'b': 'Black Bishop', 'r': 'Black Rook', 'q': 'Black Queen', 'k': 'Black King'
        }
        # Load config to get cooldown duration (config-only; no fallback defaults)
        # Agent/utils/fen_translator.py -> Agent/utils -> Agent -> root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        gs = config["game_settings"]
        self.cooldown_duration = float(gs["chess_cooldown"])

    def translate(self, enhanced_fen_full: str) -> str:
        """
        将 enhanced FEN 翻译成自然语言描述。

        输入：enhanced_fen_full（项目内常见格式：`<fen> |cooldown:...`）
        输出：多行字符串，列出双方棋子所在格子（可附带 cooldown 剩余时间）。
        """
        try:
            parts = enhanced_fen_full.split('|')
            fen = parts[0]
            cooldown_map = {} # square -> remaining_time_str
            
            if len(parts) > 1 and parts[1].startswith("cooldown:"):
                cooldown_info = parts[1].replace("cooldown:", "")
                if cooldown_info:
                    current_time = time.time()
                    for entry in cooldown_info.split(","):
                        if ":" in entry:
                            sq, start_ts_str = entry.split(":")
                            try:
                                start_ts = float(start_ts_str)
                                # Calculate remaining time
                                remaining = (start_ts + self.cooldown_duration) - current_time
                                if remaining > 0:
                                    cooldown_map[sq] = f"{remaining:.1f}s"
                            except ValueError:
                                pass

            # Fix FEN format: Realtime chess might use '*' for turn, python-chess needs 'w' or 'b'
            # Set to 'w' by default for parsing board state
            if " * " in fen:
                fen = fen.replace(" * ", " w ")

            board = chess.Board(fen)
            
            # Define display order and name mapping
            order_white = ['K', 'Q', 'R', 'N', 'B', 'P']
            order_black = ['k', 'q', 'r', 'n', 'b', 'p']
            name_map = {
                'K': 'King', 'Q': 'Queen', 'R': 'Rook', 'N': 'Knight', 'B': 'Bishop', 'P': 'Pawn',
                'k': 'King', 'q': 'Queen', 'r': 'Rook', 'n': 'Knight', 'b': 'Bishop', 'p': 'Pawn'
            }
            
            white_data = {k: [] for k in order_white}
            black_data = {k: [] for k in order_black}

            # Iterate board
            for square_idx in chess.SQUARES:
                piece = board.piece_at(square_idx)
                if piece:
                    sq_name = chess.square_name(square_idx)
                    status = sq_name
                    if sq_name in cooldown_map:
                        status += f"(Cooldown: {cooldown_map[sq_name]})"
                    
                    if piece.color == chess.WHITE:
                        white_data[piece.symbol()].append(status)
                    else:
                        black_data[piece.symbol()].append(status)

            lines = ["【Current Board Natural Language Description】"]
            
            # Build White description
            lines.append("White:")
            has_white = False
            for key in order_white:
                if white_data[key]:
                    has_white = True
                    lines.append(f"  {name_map[key]}: {', '.join(white_data[key])}")
            if not has_white:
                lines.append("  No pieces")

            # Build Black description
            lines.append("Black:")
            has_black = False
            for key in order_black:
                if black_data[key]:
                    has_black = True
                    lines.append(f"  {name_map[key]}: {', '.join(black_data[key])}")
            if not has_black:
                lines.append("  No pieces")

            # Return early with basic piece listing only (no king-relative relations)
            return "\n".join(lines)

        except Exception as e:
            return f"Failed to parse board description: {str(e)}"
