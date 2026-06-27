import chess
import random
import time

class RandomAgent:
    """
    Random Agent: Randomly selects a legal move each time.
    """
    def __init__(self, delay_s: float = 0.0):
        self.delay_s = max(0.0, float(delay_s or 0.0))
        self.last_action_source = "random"

    def get_action(self, enhanced_FEN_full, color, legal_moves):
        self.last_action_source = "random"
        move = random.choice(legal_moves) if legal_moves else None
        if self.delay_s > 0.0:
            time.sleep(self.delay_s)
        return move
