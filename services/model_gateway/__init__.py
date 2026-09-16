"""模型网关入口：get_gateway() 按 settings 的 provider 选择实现（lru_cache 单例）。

业务代码标准姿势（全项目只从这里拿网关，不直接实例化实现类）：
    from services.model_gateway import get_gateway
    gw = get_gateway()
    embs = gw.embed(["文本"])                 # -> list[Embedding(dense, sparse)]
    hits = gw.rerank(q, docs, top_n=5)        # -> list[RerankHit(index, score)] 降序
    text = gw.generate(msgs)                  # -> str；stream=True 返回迭代器；model_key="main"|"small"

provider 取值（infra.config，每个角色独立配置）：fake | openai
三个角色 provider 不同时，内部用 _RoutingGateway 按方法路由。
"""
from functools import lru_cache

from infra.config import get_settings

from .base import BaseGateway, Embedding, RerankHit
from .fake import FakeGateway
from .openai_compat import OpenAICompatGateway

FAKE = "fake"
OPENAI = "openai"

__all__ = [
    "BaseGateway",
    "Embedding",
    "FakeGateway",
    "OpenAICompatGateway",
    "RerankHit",
    "get_gateway",
]


class _RoutingGateway(BaseGateway):
    """按角色路由：embed/rerank/generate 分别委托给各自 provider 选出的后端。"""

    def __init__(self, embed_gw: BaseGateway, rerank_gw: BaseGateway, llm_gw: BaseGateway):
        self._embed_gw = embed_gw
        self._rerank_gw = rerank_gw
        self._llm_gw = llm_gw

    def embed(self, texts: list[str]) -> list[Embedding]:
        return self._embed_gw.embed(texts)

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankHit]:
        return self._rerank_gw.rerank(query, documents, top_n)

    def generate(self, messages: list[dict], *, stream: bool = False, model_key: str = "main"):
        return self._llm_gw.generate(messages, stream=stream, model_key=model_key)


@lru_cache
def get_gateway() -> BaseGateway:
    """网关单例。provider == "openai" 用 OpenAI 兼容实现，否则（含 "fake"）用 Fake。"""
    s = get_settings()
    fake = FakeGateway(dim=s.embed_dim)
    oai = OpenAICompatGateway(s)

    def pick(provider: str) -> BaseGateway:
        return oai if provider == OPENAI else fake

    embed_gw = pick(s.embed_provider)
    rerank_gw = pick(s.rerank_provider)
    llm_gw = pick(s.llm_provider)
    if embed_gw is rerank_gw is llm_gw:
        return embed_gw
    return _RoutingGateway(embed_gw, rerank_gw, llm_gw)
