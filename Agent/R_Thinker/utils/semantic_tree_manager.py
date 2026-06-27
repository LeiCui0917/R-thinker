"""语义树文本的解析、合并与路径格式化工具。"""

from __future__ import annotations

import re
from typing import Dict, Optional


# ============================================================
# 模块一：树行解析规则
# ============================================================

_NODE_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s+)?(?P<nid>0(?:\.\d+)*)\s*[:：]\s*(?P<desc>.*)\s*$"
)


# ============================================================
# 模块二：基础解析与渲染
# ============================================================

def _parse_tree_nodes(tree_text: str) -> Dict[str, str]:
    """把树文本解析成 `{node_id: desc}`。"""
    nodes: Dict[str, str] = {}
    if not tree_text:
        return nodes
    for line in tree_text.splitlines():
        m = _NODE_LINE_RE.match(line)
        if not m:
            continue
        nid = (m.group("nid") or "").strip()
        desc = (m.group("desc") or "").strip()
        if nid:
            nodes[nid] = desc
    return nodes


def _render_tree(nodes: Dict[str, str]) -> str:
    """把节点字典重新渲染成标准树文本。"""

    def sort_key(nid: str):
        try:
            return tuple(int(x) for x in nid.split("."))
        except Exception:
            return (999999,)

    lines = []
    for nid in sorted(nodes.keys(), key=sort_key):
        desc = nodes.get(nid, "")
        lines.append(f"- {nid}: {desc}".rstrip())
    return "\n".join(lines)


# ============================================================
# 模块三：树更新与路径格式化
# ============================================================

def apply_tree_patches(
    existing_tree_text: str,
    *,
    fix_nodes: Optional[Dict[str, str]] = None,
) -> str:
    """在现有树上覆盖指定节点描述。"""
    nodes = _parse_tree_nodes(existing_tree_text)
    fix_nodes = {str(k).strip(): str(v).strip() for (k, v) in (fix_nodes or {}).items() if str(k).strip()}

    if fix_nodes:
        for nid, desc in fix_nodes.items():
            nodes[nid] = desc

    return _render_tree(nodes)


def format_path_leaf_to_root(tree_text: str, node_id: str) -> str:
    """把叶节点到根节点的路径格式化成多行文本。"""
    if not tree_text or not node_id:
        return ""
    nodes = _parse_tree_nodes(tree_text)
    cur = str(node_id).strip()
    if not cur:
        return ""

    path_ids = []
    while True:
        path_ids.append(cur)
        if cur == "0":
            break
        if "." not in cur:
            break
        cur = cur.rsplit(".", 1)[0]

    lines = []
    for pid in path_ids:
        desc = nodes.get(pid)
        if desc:
            lines.append(f"{pid}: {desc}")
        else:
            lines.append(f"{pid}:")
    return "\n".join(lines)


def merge_tree(existing_tree_text: str, new_tree_text: str) -> str:
    """按 node_id 覆盖合并两棵树。"""
    nodes = _parse_tree_nodes(existing_tree_text)
    new_nodes = _parse_tree_nodes(new_tree_text)
    if new_nodes:
        nodes.update(new_nodes)
    return _render_tree(nodes)
