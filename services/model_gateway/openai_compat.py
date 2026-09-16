"""OpenAI 兼容 HTTP 协议网关实现（vLLM / TEI / 各类兼容服务通用）。

- embed：    POST {embed_base_url}/embeddings；响应含 sparse/lexical_weights 则用之，
             否则退化为字符 bigram 稀疏向量（与 FakeGateway 同一约定）
- rerank：   POST {rerank_base_url}/rerank，失败再试 /v1/rerank
- generate： POST {llm_base_url}/chat/completions，stream 用 SSE 解析 data: 行
httpx.Client 懒创建。
"""
import json
from collections.abc import Iterator

import httpx

from infra.config import Settings, get_settings

from .base import BaseGateway, Embedding, RerankHit
from .fake import _SPARSE_MOD, _stable_int, sparse_from_text


def _join(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


class OpenAICompatGateway(BaseGateway):
    """三个角色（embed/rerank/llm）各自读 Settings 中对应的 base_url/api_key/model。"""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._client: httpx.Client | None = None

    # ---------- 基础 ----------
    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=120.0)
        return self._client

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    def _parse_sparse(payload: dict | list | None, text: str) -> dict[int, float]:
        """解析服务端返回的稀疏向量；缺失/不合法时退化为文本 bigram 稀疏向量。"""
        if isinstance(payload, dict) and payload:
            out: dict[int, float] = {}
            for k, v in payload.items():
                try:
                    out[int(k)] = float(v)
                except (TypeError, ValueError):
                    out[_stable_int(str(k)) % _SPARSE_MOD] = float(v)
            if out:
                return out
        return sparse_from_text(text)

    # ---------- Embedding ----------
    def embed(self, texts: list[str]) -> list[Embedding]:
        s = self._settings
        resp = self.client.post(
            _join(s.embed_base_url, "/embeddings"),
            json={"model": s.embed_model, "input": texts, "encoding_format": "float"},
            headers=self._headers(s.embed_api_key),
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        if len(data) != len(texts):
            raise ValueError(f"embeddings 返回数量({len(data)})与输入({len(texts)})不一致")
        out = []
        for item, text in zip(data, texts):
            dense = [float(x) for x in item["embedding"]]
            sparse_raw = item.get("sparse") or item.get("lexical_weights")
            out.append(Embedding(dense=dense, sparse=self._parse_sparse(sparse_raw, text)))
        return out

    # ---------- Rerank ----------
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankHit]:
        s = self._settings
        last_err: Exception | None = None
        for path in ("/rerank", "/v1/rerank"):
            try:
                resp = self.client.post(
                    _join(s.rerank_base_url, path),
                    json={"model": s.rerank_model, "query": query, "documents": documents, "top_n": top_n},
                    headers=self._headers(s.rerank_api_key),
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                hits = [
                    RerankHit(
                        index=int(r["index"]),
                        score=float(r.get("relevance_score", r.get("score", 0.0))),
                    )
                    for r in results
                ]
                hits.sort(key=lambda h: (-h.score, h.index))
                return hits[:top_n]
            except httpx.HTTPError as exc:  # 404/连接失败等，换下一个路径重试
                last_err = exc
        raise RuntimeError(f"rerank 请求失败（已尝试 /rerank 与 /v1/rerank）: {last_err}")

    # ---------- LLM ----------
    def _model(self, model_key: str) -> str:
        s = self._settings
        if model_key == "small" and s.llm_small_model:
            return s.llm_small_model
        return s.llm_model

    def generate(
        self,
        messages: list[dict],
        *,
        stream: bool = False,
        model_key: str = "main",
    ) -> str | Iterator[str]:
        s = self._settings
        url = _join(s.llm_base_url, "/chat/completions")
        payload = {"model": self._model(model_key), "messages": messages, "stream": stream}
        if stream:
            return self._stream_generate(url, payload)
        resp = self.client.post(url, json=payload, headers=self._headers(s.llm_api_key))
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _stream_generate(self, url: str, payload: dict) -> Iterator[str]:
        s = self._settings
        with self.client.stream(
            "POST", url, json=payload, headers=self._headers(s.llm_api_key)
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
