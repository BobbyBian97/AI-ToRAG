"""确定性假网关：零依赖、结果可复现，供测试与本地开发使用。

- embed：  sha256 种子生成伪随机 dense（L2 归一化）；sparse 取字符 bigram（bigram 稳定哈希 -> 词频）
- rerank： query 与 doc 的 bigram Jaccard 相似度
- generate：输出带 [1] [2] 引用标记的确定性文本（流式为同一文本的切片）
"""
import hashlib
from collections.abc import Iterator

from .base import BaseGateway, Embedding, RerankHit

_SPARSE_MOD = 2**31  # 稀疏维度上限（与 Milvus sparse 向量习惯一致）


def _stable_int(s: str) -> int:
    """把任意字符串稳定映射为 64bit 正整数（同输入必同输出）。"""
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def _dense(text: str, dim: int) -> list[float]:
    """基于 sha256 的确定性伪随机向量，再 L2 归一化。"""
    seed = _stable_int(text)
    raw: list[float] = []
    counter = 0
    while len(raw) < dim:
        digest = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        raw.extend((b - 128) / 128.0 for b in digest)
        counter += 1
    norm = sum(x * x for x in raw[:dim]) ** 0.5 or 1.0
    return [x / norm for x in raw[:dim]]


def _bigrams(text: str) -> set[str]:
    """去空白后的字符 bigram 集合（中文友好）。"""
    t = "".join(text.split())
    return {t[i : i + 2] for i in range(len(t) - 1)}


def sparse_from_text(text: str) -> dict[int, float]:
    """字符 bigram -> 稀疏词频向量（openai_compat 的降级实现也复用此约定）。"""
    freq: dict[int, float] = {}
    for bg in _bigrams(text):
        tid = _stable_int(bg) % _SPARSE_MOD
        freq[tid] = freq.get(tid, 0.0) + 1.0
    return freq


class FakeGateway(BaseGateway):
    """确定性假实现。dim 需与 settings.embed_dim 一致（get_gateway 会传入）。"""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[Embedding]:
        return [
            Embedding(dense=_dense(t, self.dim), sparse=sparse_from_text(t))
            for t in texts
        ]

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankHit]:
        q = _bigrams(query)
        scored: list[RerankHit] = []
        for i, doc in enumerate(documents):
            d = _bigrams(doc)
            union = q | d
            score = (len(q & d) / len(union)) if union else 0.0
            scored.append(RerankHit(index=i, score=score))
        scored.sort(key=lambda h: (-h.score, h.index))
        return scored[:top_n]

    def generate(
        self,
        messages: list[dict],
        *,
        stream: bool = False,
        model_key: str = "main",
    ) -> str | Iterator[str]:
        text = self._answer(messages, model_key)
        if not stream:
            return text

        def _iter() -> Iterator[str]:
            for i in range(0, len(text), 8):
                yield text[i : i + 8]

        return _iter()

    def _answer(self, messages: list[dict], model_key: str) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        prefix = "[small]" if model_key == "small" else "[main]"
        return (
            f"{prefix} 关于「{last_user}」：知识库第 1 段给出核心要点[1]；"
            f"第 2 段补充实施细节[2]。（FakeGateway 确定性回答）"
        )
