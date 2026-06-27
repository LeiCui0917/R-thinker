import json
import logging
import os
import threading
from typing import Callable


BASE_DIR = os.path.dirname(__file__)


def load_config(base_dir: str | None = None) -> dict:
    """Load config.json from project root."""
    root = base_dir or BASE_DIR
    config_path = os.path.join(root, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    logging.info("Configuration file loaded successfully")
    return config


def load_agent_config(base_dir: str | None = None) -> dict:
    """Load Agent/agent_config.json from project root."""
    root = base_dir or BASE_DIR
    config_path = os.path.join(root, "Agent", "agent_config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    logging.info("Agent configuration file loaded successfully")
    return config


class RealtimeControllerBase:
    """Shared lifecycle skeleton for realtime controllers."""

    def __init__(self, push_callback=None):
        self.push_callback = push_callback
        self.is_running = False
        self.threads: dict[str, threading.Thread] = {}
        self.events = {
            "start": threading.Event(),
            "pause": threading.Event(),
            "end": threading.Event(),
        }

    def initialize(self) -> bool:
        raise NotImplementedError

    def _thread_specs(self) -> dict[str, tuple[Callable, tuple]]:
        raise NotImplementedError

    def _on_started(self) -> None:
        pass

    def _on_paused(self) -> None:
        pass

    def _on_resumed(self) -> None:
        pass

    def _on_stopped(self) -> None:
        pass

    def start_game(self) -> bool:
        if not self.initialize():
            return False

        self.is_running = True
        self.events["start"].set()
        self.events["pause"].clear()
        self.events["end"].clear()

        self._on_started()

        specs = self._thread_specs()
        self.threads = {}
        for name, (target, args) in specs.items():
            t = threading.Thread(target=target, args=args, daemon=True)
            self.threads[name] = t
            t.start()
        return True

    def pause_game(self):
        self.events["pause"].set()
        self._on_paused()

    def resume_game(self):
        self.events["pause"].clear()
        self._on_resumed()

    def stop_game(self):
        self.is_running = False
        self.events["end"].set()
        self._on_stopped()
