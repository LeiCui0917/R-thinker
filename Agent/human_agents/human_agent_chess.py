import threading

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
            self.pending_move = uci
        try:
            if self.move_event is not None:
                self.move_event.set()
        except Exception:
            pass

    def get_action(self, *args, **kwargs):
        self.last_action_source = "human"
        while True:
            with self._lock:
                m = self.pending_move
                self.pending_move = None
                if m is not None:
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
