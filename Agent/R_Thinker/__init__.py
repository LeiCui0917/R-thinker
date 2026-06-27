"""
R_Thinker dual-system thinking module.
Contains the Fast Thinking Module, Slow Thinking Module, and main ThinkAgent.

Framework:
- Slow Thinking Module generates prompts (prompt words/structured clues), which may be slow;
- Fast Thinking Module makes quick decisions based on prompts;
- ThinkAgent is responsible for scheduling and prompt caching, supporting multiple fast thoughts sharing one slow thought prompt.
"""

from .think_agent import ThinkAgent
from .fast_think_module import FastThinkModule
from .slow_think_module import SlowThinkModule

__all__ = [
    'ThinkAgent',
    'FastThinkModule',
    'SlowThinkModule'
]
