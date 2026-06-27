"""Maze 语义文本 grounding（兼容层空实现）。

当前 Maze 逆向语义树节点已经约定为具体坐标形式：
- 根节点：`<Side> reaches the shared goal zone at (x, y)`
- 子节点：`(a, b) to (x, y)`

因此不再需要额外的抽象坐标落地逻辑，这里只保留旧接口以兼容调用链。
"""

from __future__ import annotations

from typing import Any


def ground_maze_semantic_text(state_or_str: Any, text: str) -> str:
    """原样返回 Maze 语义文本。

    参数 `state_or_str` 继续保留，仅用于兼容旧接口；当前实现不会使用它。
    """
    _ = state_or_str
    return text or ""
