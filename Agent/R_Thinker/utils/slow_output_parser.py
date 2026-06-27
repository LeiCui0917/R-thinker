"""Slow 模块输出解析器。

职责：
1. 从 slow 模块输出中抽取新增子树节点
2. 抽取 FIX 节点修正
3. 在 guidance-only 输出时，兼容映射为根节点下的合成子节点
"""

from __future__ import annotations

import re
from typing import Any, Iterable


# ============================================================
# 模块一：正则模板
# ============================================================

_NODE_ID_RE = r"0(?:\.\d+)*"

_NEW_CHILDREN_FOR_RE = re.compile(
    rf"(?mi)^\s*(?:\*\*|#{{1,6}}|__)?\s*(?:NewChildrenFor|New\s+Children\s+for(?:\s+Node)?)\s+({_NODE_ID_RE})\s*(?:\*\*|__)?\s*[:：]\s*$"
)
_STOP_HEADING_RE = re.compile(
    r"(?mi)^\s*(?:\*\*|#{1,6}|__)?\s*(?:ChosenNode|Tree|(?:Action\s+)?Guidance|Check\s+result|NewChildrenFor|New\s+Children\s+for)\b"
)
_EXPLICIT_NODE_LINE_RE = re.compile(
    rf"^\s*(?:[-*]\s+|\d+[\.)]\s+)?(?P<nid>{_NODE_ID_RE})\s*[:：-]\s*(?P<desc>.*\S)?\s*$"
)
_NUMBERED_ITEM_RE = re.compile(r"^\s*(?:[-*]\s*)?(?P<idx>\d+)[\.)]\s*(?P<desc>.*\S)\s*$")
_FIX_LINE_RE = re.compile(
    rf"(?mi)^\s*FIX\s*(?:[:：])?\s*(?:\([^)]*\)\s*)?(?:Node\s+)?(?P<nid>{_NODE_ID_RE})\s*[:：]\s*(?P<desc>.*\S)\s*$"
)
_GUIDANCE_HEADING_RE = re.compile(r"(?mi)^\s*(?:\*\*|#{1,6}|__)?\s*Guidance\s*(?:\*\*|__)?\s*[:：]\s*$")
_SUMMARY_HEADING_RE = re.compile(r"(?mi)^\s*(?:\*\*|#{1,6}|__)?\s*Summary\s*(?:\*\*|__)?\s*[:：]\s*$")
_GENERIC_HEADING_RE = re.compile(r"(?mi)^\s*(?:\*\*|#{1,6}|__)?\s*([A-Za-z][A-Za-z0-9 _-]{0,40})\s*(?:\*\*|__)?\s*[:：]\s*$")


# ============================================================
# 模块二：基础清洗
# ============================================================

def _normalize_text(text: str) -> str:
    """统一换行，并移除 markdown 代码围栏。"""
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if "```" in text:
        text = "\n".join(line for line in text.splitlines() if not (line or "").lstrip().startswith("```"))
    return text.strip()


def _clean_desc(desc: str) -> str:
    """清理描述中的首尾杂字符和 markdown 包裹。"""
    desc = (desc or "").strip()
    if not desc:
        return ""
    desc = desc.strip().strip("：:?")
    desc = re.sub(r"^\*+|\*+$", "", desc).strip()
    return desc


# ============================================================
# 模块三：结构化节点抽取
# ============================================================

def _iter_new_children_blocks(text: str) -> Iterable[tuple[str, list[str]]]:
    """遍历 `NewChildrenFor <id>:` 代码块。"""
    matches = list(_NEW_CHILDREN_FOR_RE.finditer(text))
    for i, m in enumerate(matches):
        parent_id = (m.group(1) or "0").strip() or "0"
        start = m.end()
        end = matches[i + 1].start() if (i + 1) < len(matches) else len(text)
        yield parent_id, text[start:end].splitlines()


def _extract_node_lines(text: str) -> tuple[list[str], list[str]]:
    """抽取树节点行，返回 `(tree_lines, node_ids)`。"""
    lines: list[str] = []
    node_ids: list[str] = []
    seen_ids: set[str] = set()

    any_block = False
    for parent_id, block_lines in _iter_new_children_blocks(text):
        any_block = True
        for line in block_lines:
            if _STOP_HEADING_RE.match(line or ""):
                break
            raw = (line or "").strip()
            if not raw:
                continue

            m_explicit = _EXPLICIT_NODE_LINE_RE.match(raw)
            if m_explicit:
                nid = (m_explicit.group("nid") or "").strip()
                if nid.startswith(parent_id + "."):
                    desc = _clean_desc(m_explicit.group("desc") or "")
                    lines.append(f"- {nid}: {desc}".rstrip())
                    if nid not in seen_ids:
                        seen_ids.add(nid)
                        node_ids.append(nid)
                continue

            m_item = _NUMBERED_ITEM_RE.match(raw)
            if m_item:
                idx = (m_item.group("idx") or "").strip()
                desc = _clean_desc(m_item.group("desc") or "")
                nid = f"{parent_id}.{idx}"
                lines.append(f"- {nid}: {desc}".rstrip())
                if nid not in seen_ids:
                    seen_ids.add(nid)
                    node_ids.append(nid)

    if any_block and lines:
        return lines, node_ids

    for line in text.splitlines():
        raw = (line or "").strip()
        if not raw:
            continue
        m = _EXPLICIT_NODE_LINE_RE.match(raw)
        if not m:
            continue
        nid = (m.group("nid") or "").strip()
        if nid == "0":
            continue
        desc = _clean_desc(m.group("desc") or "")
        lines.append(f"- {nid}: {desc}".rstrip())
        if nid not in seen_ids:
            seen_ids.add(nid)
            node_ids.append(nid)

    return lines, node_ids


def _extract_fix_nodes(text: str) -> dict[str, str]:
    """抽取 `FIX <id>: ...` 修正节点。"""
    fixes: dict[str, str] = {}
    if not text:
        return fixes
    for line in text.splitlines():
        raw = (line or "").strip()
        if not raw:
            continue
        m = _FIX_LINE_RE.match(raw)
        if not m:
            continue
        nid = (m.group("nid") or "").strip()
        desc = _clean_desc(m.group("desc") or "")
        if nid and desc:
            fixes[nid] = desc
    return fixes


# ============================================================
# 模块四：guidance-only 兼容抽取
# ============================================================

def _extract_guidance_bullets(text: str) -> list[str]:
    """从 `Guidance:` 段中抽取 bullet/编号列表。"""
    if not text:
        return []

    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if _GUIDANCE_HEADING_RE.match(line or ""):
            start_idx = i + 1
            break
    if start_idx is None:
        return []

    bullets: list[str] = []
    for line in lines[start_idx:]:
        raw = (line or "").strip()
        if not raw:
            continue
        if _SUMMARY_HEADING_RE.match(raw):
            break
        if _GENERIC_HEADING_RE.match(raw) and not raw.lower().startswith("guidance"):
            break

        m_item = _NUMBERED_ITEM_RE.match(raw)
        if m_item:
            desc = _clean_desc(m_item.group("desc") or "")
            if desc:
                bullets.append(desc)
            continue

        if raw.startswith("-") or raw.startswith("*"):
            desc = _clean_desc(raw[1:].strip())
            if desc:
                bullets.append(desc)
            continue

        desc = _clean_desc(raw)
        if desc:
            bullets.append(desc)

    seen: set[str] = set()
    out: list[str] = []
    for bullet in bullets:
        if bullet in seen:
            continue
        seen.add(bullet)
        out.append(bullet)
    return out


# ============================================================
# 模块五：顶层解析入口
# ============================================================

def parse_chess_slow_output(text: str) -> dict[str, Any]:
    """解析 slow 模块输出，返回补充树和 FIX 节点。"""
    full_text = _normalize_text(text)
    if not full_text:
        return {"tree": "", "fix_nodes": {}}

    tree_lines, _ = _extract_node_lines(full_text)
    fix_nodes = _extract_fix_nodes(full_text)

    def _augment_chess_desc(desc: str) -> str:
        """补齐 Chess 节点描述中的相对 king 和路径说明。"""
        desc = (desc or "").strip()
        if not desc:
            return desc

        if not re.search(r"\b(white|black)\s+(pawn|knight|bishop|rook|queen|king)\b", desc, flags=re.IGNORECASE):
            return desc

        m_piece = re.search(r"\b(white|black)\s+(pawn|knight|bishop|rook|queen|king)\b", desc, flags=re.IGNORECASE)
        my_color = (m_piece.group(1) if m_piece else "black").strip().title()
        opp_color = "Black" if my_color == "White" else "White"

        if not re.search(r"\brelative\s+to\b", desc, flags=re.IGNORECASE):
            m_xy = re.search(r"\bat\s*\(\s*x[^,)]*\s*,\s*n[^)]*\)", desc, flags=re.IGNORECASE)
            if m_xy:
                insert_at = m_xy.end()
                desc = desc[:insert_at] + f" relative to {opp_color} King (x, n)" + desc[insert_at:]
            else:
                desc = desc + f" relative to {opp_color} King (x, n)"

        desc = re.sub(r"^\s*(white|black)\b", my_color, desc, flags=re.IGNORECASE)
        desc = re.sub(r"(?i)(?<!path\s)\bneeds\s+clearing\b", "Path needs clearing", desc)

        if re.search(r"\bPath\s+(with\s+no\s+obstruction|needs\s+clearing|does\s+not\s+affect\s+the\s+move)\b", desc, flags=re.IGNORECASE):
            desc = re.sub(r"(?<![,])\s+(Path\s+)", r", \1", desc, count=1)

        desc = re.sub(r",\s*,\s*", ", ", desc)
        desc = re.sub(r"(?i)\bpath\s+path\b", "Path", desc)
        desc = re.sub(r"(?i)\bpath\s*,\s*path\b", "Path", desc)
        return desc.strip().strip(".")

    augmented_lines: list[str] = []
    for line in tree_lines:
        m = re.match(r"^\s*-\s*(?P<nid>0(?:\.\d+)*)\s*:\s*(?P<desc>.*)\s*$", line)
        if not m:
            augmented_lines.append(line)
            continue
        nid = (m.group("nid") or "").strip()
        desc = _augment_chess_desc(m.group("desc") or "")
        augmented_lines.append(f"- {nid}: {desc}".rstrip())
    tree_lines = augmented_lines

    if fix_nodes:
        fix_nodes = {k: _augment_chess_desc(v) for k, v in fix_nodes.items()}

    if not tree_lines:
        bullets = _extract_guidance_bullets(full_text)
        if bullets:
            tree_lines = [f"- 0.{i + 1}: {b}" for i, b in enumerate(bullets)]

    return {
        "tree": "\n".join(tree_lines).strip(),
        "fix_nodes": fix_nodes,
    }
