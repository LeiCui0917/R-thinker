import random
import time

class RandomAgent:
    """
    Random Agent (Maze, race-only): Randomly selects a legal target junction.
    """
    def __init__(self, delay_s: float = 0.0):
        self.delay_s = max(0.0, float(delay_s or 0.0))
        self.last_action_source = "random"

    def _extract_target(self, opt):
        if isinstance(opt, (tuple, list)) and len(opt) == 2:
            try:
                return (int(opt[0]), int(opt[1]))
            except Exception:
                return None
        return None

    def get_action(self, frame=None, role=None, legal_moves=None):
        self.last_action_source = "random"
        if not legal_moves:
            return None

        # Coordinate-only behavior: legal_moves are junction options as (x, y).
        targets = []
        for opt in legal_moves:
            t = self._extract_target(opt)
            if t is not None:
                targets.append(t)
        if not targets:
            return None
        move = random.choice(targets)
        if self.delay_s > 0.0:
            time.sleep(self.delay_s)
        return move
