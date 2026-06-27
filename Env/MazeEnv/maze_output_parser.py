"""Env/MazeEnv/maze_output_parser.py

【模块：输出解析器（Maze）】

职责：
- 将 LLM 输出文本解析成 Maze 目标路口坐标 `(x, y)`（字符串级解析，不依赖地图合法性）。

设计约束：
- 仅做“提取/清洗/格式校验”，不做路径可达性或合法动作校验。
- 保持极简：只提供三步管线（取最后一句 → 提取目标坐标 → 校验格式）。
"""

import re
import unicodedata

class MazeOutputParser:
    """极简版输出解析器：只保留 3 个函数。

    1) get_last_sentence: 提取 LLM 输出的最后一句话
    2) find_target_junction_in_sentence: 从这句话里提取目标路口坐标
    3) is_valid_target_junction: 判断目标坐标格式是否合法
    """

    _JUNCTION_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")

    # ---------------- 3-step parsing API ----------------
    # 1) get_last_sentence
    # 2) find_target_junction_in_sentence
    # 3) is_valid_target_junction

    def get_last_sentence(self, text: str) -> str:
        """提取输出的“最后一句话”。

        规则（偏鲁棒）：
        - 先取最后一个非空行，并按句末标点切分成若干“句子”。
        - 从后往前找：优先返回“包含可解析坐标”的最后一句。
        - 如果所有句子都不包含坐标，则退化为：返回最后一句。
        """
        norm = unicodedata.normalize('NFKC', text or '')
        lines = [ln.strip() for ln in norm.splitlines() if ln.strip()]
        if not lines:
            return ''
        last_line = lines[-1]
        parts = [p.strip() for p in re.split(r'[。！？.!?]+', last_line) if p.strip()]
        if not parts:
            return last_line

        # 从后往前找，优先返回能解析出坐标的最后一句。
        for candidate in reversed(parts):
            target = self.find_target_junction_in_sentence(candidate)
            if self.is_valid_target_junction(target):
                return candidate
        return parts[-1]

    # ---------------- Junction-target parsing API ----------------

    def find_target_junction_in_sentence(self, sentence: str):
        """从一句话中提取目标路口坐标（取最后一个有效匹配）。

        兼容常见写法：
        - `(13,7)` / `(13, 7)`
        - `target: (13, 7)`
        - `IntentPath: (x0,y0) -> (x1,y1) -> (x2,y2)`

        规则：
        - 若检测到 IntentPath：
          - 旧格式（首节点是坐标）返回第二个坐标（通常表示下一目标）；
          - 新格式（首节点非坐标）返回第一个坐标节点。
        - 否则回退为：返回句子里最后一个坐标。
        """
        if not sentence:
            return None
        norm = unicodedata.normalize('NFKC', sentence)
        cleaned = ''.join((' ' if unicodedata.category(ch)[0] == 'C' else ch) for ch in norm)

        # 优先按 IntentPath 结构化解析。
        nodes = self._parse_intent_nodes(cleaned)
        if nodes:
            first_coord = self._parse_coord_token(nodes[0])

            coord_nodes = []
            for node in nodes:
                parsed = self._parse_coord_token(node)
                if parsed is not None:
                    coord_nodes.append(parsed)

            if coord_nodes:
                if first_coord is not None:
                    return coord_nodes[1] if len(coord_nodes) >= 2 else coord_nodes[0]
                return coord_nodes[0]

        # 通用回退：取最后一个坐标匹配。
        matches = list(self._JUNCTION_RE.finditer(cleaned))
        if not matches:
            return None

        m = matches[-1]
        try:
            x = int(m.group(1))
            y = int(m.group(2))
        except Exception:
            return None
        return (x, y)

    def is_valid_target_junction(self, target) -> bool:
        """判断目标路口坐标格式是否合法。

        约束：
        - 只做“字符串级”产物校验（不依赖地图/局面）。
        - 仅接受 `tuple[int, int]`。
        """
        return (
            isinstance(target, tuple)
            and len(target) == 2
            and isinstance(target[0], int)
            and isinstance(target[1], int)
        )

    def _parse_coord_token(self, token: str):
        if not isinstance(token, str):
            return None
        m = self._JUNCTION_RE.fullmatch(token.strip())
        if not m:
            return None
        try:
            return (int(m.group(1)), int(m.group(2)))
        except Exception:
            return None

    def _parse_intent_nodes(self, text: str) -> list[str]:
        lower = text.lower()
        idx = lower.find("intentpath")
        if idx < 0:
            return []
        sub = text[idx:]
        parts = sub.split(":", 1)
        body = parts[1] if len(parts) == 2 else parts[0]
        return [p.strip() for p in body.split("->") if p.strip()]