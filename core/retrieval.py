"""混合检索与融合（设计书 §4.2 在线问答流程第 2~4 步）。

流水线：embed(query) -> search_hybrid（dense/sparse 各 top50，kb/ACL 过滤已在召回期下推）
        -> rrf_fuse 融合候选 30 -> PG 补全 chunk 元数据（元数据事实源）-> rerank 精排取 top_k。

实现约定：本模块通过「模块属性」在调用时访问 infra.milvus / services.model_gateway 的函数，
以便单元测试用 monkeypatch 替换（不要改成 from x import fn 的静态绑定）。
"""
import logging
from dataclasses import dataclass, replace

from sqlalchemy import select
from sqlalchemy.orm import Session

import infra.milvus
from infra.models import Chunk, Document
from services import model_gateway

logger = logging.getLogger(__name__)

RECALL_TOP_K = 50  # 每路召回条数（设计书 §4.2：dense(top50) ∥ sparse(top50)）
RRF_K = 60         # RRF 平滑常数
FUSE_TOP_N = 30    # RRF 融合后的候选数


@dataclass
class RetrievedChunk:
    """补全元数据后的检索结果（供 prompt 组装 / refs 事件 / message.refs 落库）。"""

    chunk_id: int
    doc_id: int
    doc_title: str
    page_no: int | None
    score: float
    content: str


def rrf_fuse(rank_lists: list[list[int]], k: int = RRF_K, top_n: int = FUSE_TOP_N) -> list[int]:
    """Reciprocal Rank Fusion：融合多路按序 chunk_id 排名，返回去重后的 id 列表。

    得分：score(c) = Σ 1/(k + rank)，rank 为 c 在各路中的 1 起始名次；
    并列时按 chunk_id 升序，保证输出确定可测。纯函数。
    """
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, chunk_id in enumerate(ranks, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + pos)
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return ordered[:top_n]


def retrieve(
    db: Session,
    query: str,
    kb_ids: list[int],
    top_k: int = 8,
    doc_ids: list[int] | None = None,
    use_rerank: bool = True,
) -> list[RetrievedChunk]:
    """混合检索 + RRF 融合 + rerank 精排，返回可直接用于生成上下文的块列表。

    - doc_ids：F4 ACL 下推的可见文档集（None 表示不过滤），原样透传给 search_hybrid；
    - rerank 异常时降级为 RRF 融合序（记 log 不抛）；
    - 候选 chunk 若在 PG 中缺行（如向量库脏数据/软删未清理）则直接丢弃。
    """
    gw = model_gateway.get_gateway()
    emb = gw.embed([query])[0]

    # 召回（kb/ACL 过滤在 Milvus 侧完成，绝不做检索后过滤）
    dense_hits, sparse_hits = infra.milvus.search_hybrid(
        kb_ids, emb.dense, emb.sparse, top_k=RECALL_TOP_K, doc_ids=doc_ids
    )

    # 融合前的兜底分数：同一 chunk 取两路中的最高分
    score_map: dict[int, float] = {}
    for hits in (dense_hits, sparse_hits):
        for h in hits:
            score_map[h.chunk_id] = max(score_map.get(h.chunk_id, h.score), h.score)

    candidates = rrf_fuse(
        [[h.chunk_id for h in dense_hits], [h.chunk_id for h in sparse_hits]],
        k=RRF_K,
        top_n=FUSE_TOP_N,
    )
    if not candidates:
        return []

    # PG 元数据事实源：批量取 content/page_no，join document 取标题；缺行候选丢弃
    rows = db.execute(
        select(Chunk.id, Chunk.doc_id, Chunk.page_no, Chunk.content, Document.title)
        .join(Document, Document.id == Chunk.doc_id)
        .where(Chunk.id.in_(candidates))
    ).all()
    by_id = {row.id: row for row in rows}
    chunks = [
        RetrievedChunk(
            chunk_id=cid,
            doc_id=by_id[cid].doc_id,
            doc_title=by_id[cid].title or f"文档 #{by_id[cid].doc_id}",
            page_no=by_id[cid].page_no,
            score=score_map.get(cid, 0.0),
            content=by_id[cid].content,
        )
        for cid in candidates
        if cid in by_id
    ]
    if not chunks or not use_rerank:
        return chunks[:top_k]

    # 精排：rerank 打分覆盖融合分数，按分数降序取 top_k
    try:
        rerank_hits = gw.rerank(query, [c.content for c in chunks], top_n=top_k)
    except Exception:
        logger.warning("rerank 调用失败，降级使用 RRF 融合序", exc_info=True)
        return chunks[:top_k]
    ranked = [
        replace(chunks[h.index], score=h.score)
        for h in rerank_hits
        if 0 <= h.index < len(chunks)
    ]
    if not ranked:  # rerank 返回空/全部越界时兜底
        return chunks[:top_k]
    return ranked
