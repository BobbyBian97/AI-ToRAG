"""切分层（F2 流水线第二步）：ParsedDoc -> ChunkDraft 列表。

策略（设计书 §4.1，标题感知递归切分 + small-to-big）：
- 父块 = 每个「叶子章节」（标题路径相同且连续的 block 组）的全文聚合，
  不切分、不限长，父块不进向量库，仅供检索命中子块后拼接生成上下文；
- 子块 = 章节内文本按目标 512 token 递归切分（段落 -> 行 -> 句子 -> 硬窗口
  逐级降级），相邻块重叠约 64 token；重叠取上一块尾部的句子/行边界，
  并计入 512 预算（保证任何子块都不超目标长度）；
- 表格块（is_table）永不切分，超长也整块（保证表格语义完整）。

纯函数、无 IO，方便单测。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from core.parsing import ParsedBlock, ParsedDoc

# 切分参数（设计书 §4.1：目标 512 / 重叠 64）
CHUNK_TARGET_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

# 无自然边界文本（无换行、无句子标点）的硬切窗口，与重叠同量级，
# 保证装箱后仍可在硬窗口边界取到重叠
_HARD_WINDOW_TOKENS = CHUNK_OVERLAP_TOKENS

# 重叠段的最小 token 数：太碎的重叠没有语义价值，宁可为空
_MIN_OVERLAP_TOKENS = 8

# 句边界（中英标点），lookbehind 保证分隔符留在句尾
_SENTENCE_DELIM_RE = re.compile(r"(?<=[。！？!?；;…])")


@dataclass
class ChunkDraft:
    """切分草稿：落库（services.worker.tasks）时由 seq/parent_seq 回填 id 关系。"""

    content: str
    token_count: int
    seq: int  # 文档内从 0 连续递增（父块/子块统一编号）
    headings: str  # " > " 连接的标题路径字符串（顶层在前），无标题为 ""
    page_no: int | None
    parent_seq: int | None  # 所属父块（章节块）的 seq；父块自身为 None


def approx_token_len(text: str) -> int:
    """近似 token 数（中文优先）：CJK/全角等非 ASCII 每字计 1，
    可打印 ASCII 连续段每 4 字符计 1（向上取整），换行等空白不计。
    """
    total = 0
    run = 0  # 当前可打印 ASCII 连续段长度
    for ch in text:
        if " " <= ch <= "~":
            run += 1
        else:
            if run:
                total += (run + 3) // 4
                run = 0
            if not ch.isspace():
                total += 1
    if run:
        total += (run + 3) // 4
    return total


def chunk_document(doc: ParsedDoc) -> list[ChunkDraft]:
    """标题感知切分主入口：每个叶子章节产出 1 个父块 + N 个子块，seq 全文档连续。"""
    drafts: list[ChunkDraft] = []
    seq = 0
    # 过滤空白块，避免产生空 chunk / 空章节
    valid_blocks = [b for b in doc.blocks if b.text and b.text.strip()]
    for path, blocks in _group_sections(valid_blocks):
        headings_str = " > ".join(path)
        # 父块：章节全文聚合，不切分、不限长（不进向量库）
        parent_text = "\n\n".join(b.text for b in blocks)
        parent_seq = seq
        drafts.append(
            ChunkDraft(
                content=parent_text,
                token_count=approx_token_len(parent_text),
                seq=seq,
                headings=headings_str,
                page_no=blocks[0].page_no,
                parent_seq=None,
            )
        )
        seq += 1
        for b in blocks:
            # 表格永不切分；其余文本递归切分
            pieces = [b.text] if b.is_table else _split_text(b.text)
            for piece in pieces:
                drafts.append(
                    ChunkDraft(
                        content=piece,
                        token_count=approx_token_len(piece),
                        seq=seq,
                        headings=headings_str,
                        page_no=b.page_no,
                        parent_seq=parent_seq,
                    )
                )
                seq += 1
    return drafts


def _group_sections(blocks: list[ParsedBlock]) -> list[tuple[tuple[str, ...], list[ParsedBlock]]]:
    """把 block 流按标题路径分组：路径相同且连续的 block 构成一个叶子章节。

    例：h1 下的引言（路径 ["指南"]）与 h2 正文（路径 ["指南","安装"]）分属两组，
    各自成父块 —— 标题层级即章节边界。
    """
    groups: list[tuple[tuple[str, ...], list[ParsedBlock]]] = []
    for b in blocks:
        key = tuple(b.headings)
        if groups and groups[-1][0] == key:
            groups[-1][1].append(b)
        else:
            groups.append((key, [b]))
    return groups


# --------------------------------------------------------------------------
# 递归切分（纯文本块）
# --------------------------------------------------------------------------
def _split_text(
    text: str,
    target: int = CHUNK_TARGET_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """把一段文本切成若干 ≤ target token 的块，相邻块重叠约 overlap token。

    两步：原子化（段落/行/句子/硬窗口，每个原子 ≤ target）+ 贪心装箱
    （装不下就收口，并以尾部原子作为下一块的重叠前缀）。
    """
    atoms = _atomize(text, target)
    pieces: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for atom in atoms:
        n = approx_token_len(atom)
        if cur and cur_len + n > target:
            pieces.append("\n".join(cur))
            # 重叠预算 = target - 下一个原子的大小，保证「重叠 + 新内容」仍 ≤ target
            allowed = max(0, target - n)
            cur = _tail_units(cur, min(overlap, allowed))
            cur_len = sum(approx_token_len(u) for u in cur)
        cur.append(atom)
        cur_len += n
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def _atomize(text: str, max_tokens: int) -> list[str]:
    """递归降级切原子单元：段落 -> 行 -> 句子 -> 硬窗口，每个单元 ≤ max_tokens。"""
    atoms: list[str] = []
    for para in text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n"):
        if approx_token_len(para) <= max_tokens:
            atoms.append(para)
            continue
        for line in para.split("\n"):
            if approx_token_len(line) <= max_tokens:
                atoms.append(line)
                continue
            for sent in _split_sentences(line):
                if approx_token_len(sent) <= max_tokens:
                    atoms.append(sent)
                else:
                    # 句子本身超长（无标点长文本）：按硬窗口切，窗口与重叠同量级
                    atoms.extend(_hard_windows(sent))
    return atoms


def _split_sentences(line: str) -> list[str]:
    """按中英文句尾标点切句（分隔符保留在句尾）。"""
    parts = _SENTENCE_DELIM_RE.split(line)
    return [p for p in parts if p and p.strip()]


def _hard_windows(sent: str, window_tokens: int = _HARD_WINDOW_TOKENS) -> list[str]:
    """无可读边界时的兜底：按 token 数硬切成连续窗口（每字最多计 1 token，故窗口不会超限）。"""
    windows: list[str] = []
    buf = ""
    for ch in sent:
        buf += ch
        if approx_token_len(buf) >= window_tokens:
            windows.append(buf)
            buf = ""
    if buf.strip():
        windows.append(buf)
    return windows


def _tail_units(units: list[str], limit: int) -> list[str]:
    """取 units 尾部、总量 ≤ limit token 的重叠段（优先整句/整行边界）。

    尾部单元自身超限时，退化为字符级截取该单元的尾部（边界极端情况的兜底）。
    """
    if limit <= 0:
        return []
    tail: list[str] = []
    total = 0
    for u in reversed(units):
        n = approx_token_len(u)
        if total + n > limit:
            if not tail:
                sub = _tail_substring(u, limit)
                if sub:
                    tail.insert(0, sub)
            break
        tail.insert(0, u)
        total += n
    return tail


def _tail_substring(text: str, limit: int) -> str:
    """从文本尾部向前截取不超过 limit token 的子串；太碎（< 最小重叠）则返回空。"""
    sub = ""
    for ch in reversed(text):
        cand = ch + sub
        if approx_token_len(cand) > limit:
            break
        sub = cand
    return sub if approx_token_len(sub) >= min(_MIN_OVERLAP_TOKENS, limit) else ""
