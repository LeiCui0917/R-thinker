import threading
import re

class HumanAgent:
    def __init__(self, move_event=None, stop_event=None, wait_timeout_s: float = 0.25, **kwargs):
        self.pending_move = None
        self._lock = threading.Lock()
        self.move_event = move_event
        self.stop_event = stop_event
        self.wait_timeout_s = max(0.05, float(wait_timeout_s or 0.25))
        self.last_action_source = "human"

    def set_pending_move(self, uci: str):
        with self._lock:
            if isinstance(uci, (tuple, list)) and len(uci) == 2:
                try:
                    self.pending_move = (int(uci[0]), int(uci[1]))
                    try:
                        if self.move_event is not None:
                            self.move_event.set()
                    except Exception:
                        pass
                    return
                except Exception:
                    self.pending_move = None
                    return

            m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", str(uci or ""))
            if m:
                self.pending_move = (int(m.group(1)), int(m.group(2)))
            else:
                self.pending_move = None
        try:
            if self.move_event is not None:
                self.move_event.set()
        except Exception:
            pass

    def get_action(self, frame=None, role=None, legal_moves=None, **kwargs):
        self.last_action_source = "human"
        while True:
            with self._lock:
                m = self.pending_move
                self.pending_move = None
                if m is not None:
                    # Maze uses coordinate targets as (x, y).
                    return m

            try:
                if self.stop_event is not None and self.stop_event.is_set():
                    return None
            except Exception:
                pass

            if self.move_event is None:
                return None

            try:
                self.move_event.clear()
                self.move_event.wait(self.wait_timeout_s)
            except Exception:
                return None
