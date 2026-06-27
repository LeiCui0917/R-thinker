"""Agent/utils/chess_output_parser.py

【模块：输出解析器（Chess）】

职责：
- 将 LLM 输出文本解析成 Chess 可执行的 UCI 走法（字符串级解析，不依赖棋盘合法性）。

设计约束：
- 仅做“提取/清洗/格式校验”，不做局面校验（合法性由上层环境/规则决定）。
- 保持极简：只提供三步管线（取最后一句 → 提取走法 → 校验格式）。
"""

import re
import unicodedata


class ChessOutputParser:
    """极简版输出解析器：只保留 3 个函数。

    1) get_last_sentence: 提取 LLM 输出的最后一句话
    2) find_uci_in_sentence: 从这句话里提取 UCI 走法
    3) is_valid_uci: 判断 UCI 字符串格式是否合法
    """

    def get_last_sentence(self, text: str) -> str:
        """提取输出的“最后一句话”。

        规则（偏鲁棒）：
        - 先取最后一个非空行，并按句末标点切分成若干“句子”。
        - 从后往前找：优先返回“包含可解析 UCI”的最后一句。
        - 如果所有句子都不包含 UCI，则退化为：返回最后一句。
        """
        norm = unicodedata.normalize('NFKC', text or '')
        lines = [ln.strip() for ln in norm.splitlines() if ln.strip()]
        if not lines:
            return ''
        last_line = lines[-1]
        parts = [p.strip() for p in re.split(r'[。！？.!?]+', last_line) if p.strip()]
        if not parts:
            return last_line

        # 从后往前找，优先返回能解析出 UCI 的最后一句。
        for candidate in reversed(parts):
            if self.find_uci_in_sentence(candidate):
                return candidate
        return parts[-1]

    def find_uci_in_sentence(self, sentence: str) -> str:
        """从一句话中提取 UCI 走法（取最后一个匹配）。

        兼容一些常见“非标准写法/噪声”：
        - 分隔符：`e2-e4` / `e2 e4` / `e2→e4` / `e2 to e4`
        - 升变：`e7e8q` / `e7e8=Q` / `e7-e8=Q`
        - 额外符号：`e2e4+`、括号、引号、中文标点等（会被清洗/忽略）
        """
        if not sentence:
            return ''
        norm = unicodedata.normalize('NFKC', sentence)
        # 1) 统一大小写；2) 控制字符替换为空格；3) 常见连字符/箭头归一化为空格，方便分段匹配
        lowered = norm.lower()
        cleaned = ''.join((' ' if unicodedata.category(ch)[0] == 'C' else ch) for ch in lowered)
        cleaned = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2212\-→⇒➡⟶⟹]+', ' ', cleaned)

        # 先尝试直接 UCI（最常见/最可靠）：e2e4 / e7e8q
        direct_matches = list(re.finditer(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', cleaned))
        if direct_matches:
            cand = direct_matches[-1].group(1).lower()
            return cand if self.is_valid_uci(cand) else ''

        # 再尝试“两个格子被分隔开”的写法：
        # - e2 e4
        # - e2 to e4 / from e2 to e4
        # - e2 → e4（箭头已在上面被归一化为空格，所以这里主要处理 'to/到/至' 等连接词）
        # - 升变：e7 to e8 = q / e7 e8 q / e7 to e8 promote to queen
        # 更直接地匹配 “from <sq> to <sq>” / “<sq> to <sq>”
        # 这里允许连接词存在于两个格子之间。
        # Promotion variants supported here:
        # - e7 to e8 = q
        # - e7 to e8q   (destination square + promo letter attached)
        to_pattern = re.compile(
            r'\b(?:from\s+)?(?P<frm>[a-h][1-8])\b\s*'
            r'(?:to|2|→|到|至)\s*'
            r'(?P<to>[a-h][1-8])(?:(?:\s*=\s*)?(?P<promo>[qrbn]))?\b'
        )
        to_matches = list(to_pattern.finditer(cleaned))
        if to_matches:
            last = to_matches[-1]
            frm = last.group('frm')
            to = last.group('to')
            promo = (last.group('promo') or '').lower()
            tail = cleaned[last.end():]

            if not promo:
                promo_m = re.search(
                    r'(?:=\s*|promot\w*\s*(?:to\s*)?)'
                    r'(q|r|b|n|queen|rook|bishop|knight)\b',
                    tail,
                )
                if promo_m:
                    token = promo_m.group(1)
                    promo = {
                        'q': 'q', 'queen': 'q',
                        'r': 'r', 'rook': 'r',
                        'b': 'b', 'bishop': 'b',
                        'n': 'n', 'knight': 'n',
                    }.get(token, '')
                else:
                    # 如果目的地在 1/8 排、并且句子里明确提到 pawn 或 promote，默认升变为 q
                    if to[1] in ('1', '8') and (('pawn' in cleaned) or ('promot' in cleaned) or ('=' in cleaned)):
                        promo = 'q'

            cand = f'{frm}{to}{promo}'.lower()
            return cand if self.is_valid_uci(cand) else ''

        # 最后再尝试“纯两个格子挨着/空格分隔”的写法：e2 e4 / e7 e8 q
        sep_pattern = re.compile(
            r'\b([a-h][1-8])\b\s+([a-h][1-8])(?:(?:\s*=\s*)?([qrbn]))?\b'
        )
        sep_matches = list(sep_pattern.finditer(cleaned))
        if not sep_matches:
            return ''

        last = sep_matches[-1]
        frm = last.group(1)
        to = last.group(2)
        promo = (last.group(3) or '')
        cand = f'{frm}{to}{promo}'.lower()
        return cand if self.is_valid_uci(cand) else ''

    def is_valid_uci(self, move_str: str) -> bool:
        """判断 UCI 字符串格式是否合法。

        约束：
        - 只做“字符串级”校验（不依赖棋盘局面）。
        - 支持升变 `e7e8q`，并限制：
          - 升变目标必须在 1/8 排
          - 升变来源必须在 2/7 排（避免 `e3e8q` 之类明显错误）
        - 起点终点不能相同（不接受空走）。
        """
        if not move_str:
            return False
        m = move_str.strip().lower()
        if len(m) == 4:
            if not (m[0] in 'abcdefgh' and m[1] in '12345678' and
                    m[2] in 'abcdefgh' and m[3] in '12345678'):
                return False
            return m[:2] != m[2:4]
        if len(m) == 5 and m[4] in 'qrbn':
            if not (m[0] in 'abcdefgh' and m[1] in '12345678' and
                    m[2] in 'abcdefgh' and m[3] in '12345678'):
                return False
            if m[:2] == m[2:4]:
                return False
            # 升变只能落在 1/8 排
            if m[3] not in ('1', '8'):
                return False
            # 进一步约束：升变来源只能在 2/7 排
            if m[3] == '8' and m[1] != '7':
                return False
            if m[3] == '1' and m[1] != '2':
                return False
            return True
        return False
