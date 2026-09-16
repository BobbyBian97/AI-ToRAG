"""Milvus 封装（pymilvus MilvusClient 风格：懒连接 + 幂等）。

共享契约（F2 写入 / F3 检索 / F4 权限过滤）：
- ensure_collection()            建 collection + 索引（幂等）
- insert_chunks(rows)            rows 键固定：chunk_id/kb_id/doc_id/dense(list[float])/sparse(dict[int,float])/text
- delete_chunks(chunk_ids)       按 chunk 主键批量删除
- search_hybrid(kb_ids, dense, sparse, top_k, doc_ids=None)
                                 dense/sparse 各检索 top_k，返回 (dense_hits, sparse_hits)
                                 kb_ids / doc_ids 在召回阶段做标量过滤（设计书 §4.2：绝不能检索后再过滤）
- Hit                            dataclass{chunk_id: int, score: float}

注意：本模块所有函数首次调用才会真正连接 Milvus；单元测试一律 mock get_client。
"""
from dataclasses import dataclass
from functools import lru_cache

from pymilvus import DataType, MilvusClient

from infra.config import get_settings


@dataclass
class Hit:
    """检索单条命中。score：dense 分支为相似度（越大越好），sparse 分支为稀疏得分。"""

    chunk_id: int
    score: float


@lru_cache
def get_client() -> MilvusClient:
    """MilvusClient 单例（懒连接）。"""
    return MilvusClient(uri=get_settings().milvus_uri)


def ensure_collection() -> None:
    """建 chunk collection + 索引（幂等：已存在则直接返回）。

    fields：
      chunk_id  int64   主键（对应 PG chunk.id）
      kb_id     int64   知识库过滤
      doc_id    int64   文档过滤（设计书 §7 未列，但 §4.2 召回期权限过滤必须有它——有意补充）
      dense     float16_vector, dim=embed_dim（HNSW M=16 efConstruction=200, IP）
      sparse    sparse_float_vector（SPARSE_INVERTED_INDEX）
      text      varchar 仅调试用
    """
    s = get_settings()
    client = get_client()
    if client.has_collection(s.milvus_collection):
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.INT64, is_primary=True)
    schema.add_field("kb_id", DataType.INT64)
    schema.add_field("doc_id", DataType.INT64)
    schema.add_field("dense", DataType.FLOAT16_VECTOR, dim=s.embed_dim)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True)

    # 关于 sparse 与 BM25 的选型说明（TODO）：
    # 当前契约是「模型侧产出 sparse」（BGE-M3 lexical weights / FakeGateway 字符 bigram），
    # 即 insert_chunks 显式写入 sparse、search_hybrid 用 sparse 向量查询，
    # 因此 sparse 索引用 SPARSE_INVERTED_INDEX + IP 度量（对模型产出的稀疏向量通用）。
    # 若后续切换为 Milvus 原生全文检索（BM25 Function），需改为：
    #   from pymilvus import Function, FunctionType
    #   schema.add_function(Function(
    #       name="bm25", function_type=FunctionType.BM25,
    #       input_field_names=["text"], output_field_names=["sparse"]))
    # 届时 sparse 由 Milvus 从 text 自动计算：insert_chunks 必须去掉 sparse 键，
    # search_hybrid 的 sparse 分支改为按查询原文检索（data=[query_text]），
    # 索引 metric_type 相应改为 "BM25"。两种模式不可混用。
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense",
        index_type="HNSW",
        metric_type="IP",
        params={"M": 16, "efConstruction": 200},
    )
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )
    client.create_collection(
        collection_name=s.milvus_collection,
        schema=schema,
        index_params=index_params,
    )


def insert_chunks(rows: list[dict]) -> None:
    """批量写入 chunk 向量（F2 流水线末段调用）。

    rows 每项键固定：chunk_id:int, kb_id:int, doc_id:int,
                     dense:list[float], sparse:dict[int,float], text:str
    """
    s = get_settings()
    entities = [
        {
            "chunk_id": int(r["chunk_id"]),
            "kb_id": int(r["kb_id"]),
            "doc_id": int(r["doc_id"]),
            "dense": [float(x) for x in r["dense"]],
            "sparse": {int(k): float(v) for k, v in r["sparse"].items()},
            "text": str(r["text"]),
        }
        for r in rows
    ]
    if not entities:
        return
    ensure_collection()
    get_client().insert(collection_name=s.milvus_collection, data=entities)


def delete_chunks(chunk_ids: list[int]) -> None:
    """按 chunk 主键批量删除（文档软删后的异步清理，TASK_PURGE 使用）。"""
    if not chunk_ids:
        return
    s = get_settings()
    ids = [int(i) for i in chunk_ids]
    get_client().delete(collection_name=s.milvus_collection, filter=f"chunk_id in {ids}")


def _build_filter(kb_ids: list[int], doc_ids: list[int] | None) -> str | None:
    """构造召回期标量过滤表达式；返回 None 表示无可行集（调用方直接返回空结果）。"""
    if not kb_ids:
        return None
    expr = f"kb_id in {sorted(int(k) for k in kb_ids)}"
    if doc_ids is not None:
        if len(doc_ids) == 0:
            return None
        expr += f" and doc_id in {sorted(int(d) for d in doc_ids)}"
    return expr


def _parse_hits(result: list) -> list[Hit]:
    """把 MilvusClient.search 的返回（list[list[dict]]）转成 Hit 列表。"""
    hits = []
    for row in (result[0] if result else []):
        hits.append(Hit(chunk_id=int(row["id"]), score=float(row["distance"])))
    return hits


def search_hybrid(
    kb_ids: list[int],
    dense: list[float],
    sparse: dict[int, float],
    top_k: int,
    doc_ids: list[int] | None = None,
) -> tuple[list[Hit], list[Hit]]:
    """混合检索：dense 与 sparse 各自检索 top_k，返回 (dense_hits, sparse_hits)。

    过滤语义（设计书 §4.2，必须召回阶段过滤）：
      - kb_ids：限定知识库，空列表 => 直接返回 ([], [])；
      - doc_ids 非 None 时按用户可见文档集过滤（F4 ACL 下推），空列表 => ([], [])；
        为 None 表示不过滤文档。
    """
    s = get_settings()
    expr = _build_filter(kb_ids, doc_ids)
    if expr is None:
        return [], []

    client = get_client()
    dense_result = client.search(
        collection_name=s.milvus_collection,
        data=[[float(x) for x in dense]],
        anns_field="dense",
        limit=top_k,
        filter=expr,
        search_params={"metric_type": "IP", "params": {"ef": max(64, top_k)}},
    )
    sparse_result = client.search(
        collection_name=s.milvus_collection,
        data=[{int(k): float(v) for k, v in sparse.items()}],
        anns_field="sparse",
        limit=top_k,
        filter=expr,
        search_params={"metric_type": "IP"},
    )
    return _parse_hits(dense_result), _parse_hits(sparse_result)
