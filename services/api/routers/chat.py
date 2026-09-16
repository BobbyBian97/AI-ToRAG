"""F3 问答路由：POST /api/v1/chat（SSE 流式，设计书 §8 契约）。

事件序列（正常流以 refs -> done 结束）：
  event: delta  data: {"text": "..."}                                  # 生成增量，0..n 条
  event: refs   data: [{"chunk_id","doc_id","doc_title","page_no","score"}, ...]
  event: done   data: {"conversation_id": int, "message_id": int, "latency_ms": int}
生成器内异常时以 error 事件收尾：
  event: error  data: {"detail": "..."}
"""
import json
import logging
import time
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.prompt import NO_RESULT_TEXT, build_messages, rewrite_standalone_question
from core.retrieval import retrieve
from infra.models import Conversation, Message
from services import model_gateway
from services.api.deps import UserCtx, get_current_user, get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])

MAX_KB_IDS = 10   # 单次问答可跨的知识库数上限
MAX_QUERY_LEN = 4000
ANSWER_TOP_K = 8  # 进入生成上下文的块数（设计书 §4.2：精排取 top 5~8）
TITLE_LEN = 30    # 新会话标题 = 首问前 30 字

DB = Annotated[Session, Depends(get_db)]
User = Annotated[UserCtx, Depends(get_current_user)]


class ChatRequest(BaseModel):
    """POST /api/v1/chat 请求体（kb_ids 空/超限、query 空 => 422）。"""

    conversation_id: int | None = None
    kb_ids: list[int] = Field(min_length=1, max_length=MAX_KB_IDS)
    query: str = Field(min_length=1, max_length=MAX_QUERY_LEN)


def _sse(event: str, data) -> str:
    """格式化一条 SSE 事件（event + data 两行，空行结尾）。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _refs_payload(hits) -> list[dict]:
    """refs 事件 / message.refs 共用的引用列表结构。"""
    return [
        {
            "chunk_id": h.chunk_id,
            "doc_id": h.doc_id,
            "doc_title": h.doc_title,
            "page_no": h.page_no,
            "score": h.score,
        }
        for h in hits
    ]


@router.post("/chat", summary="知识库问答（SSE 流式，以 refs/done 事件收尾）")
def chat(req: ChatRequest, db: DB, user: User) -> StreamingResponse:
    # 1) 会话：无 conversation_id 则以首问前 30 字为标题新建
    if req.conversation_id is None:
        conv = Conversation(user_id=user.user_id, title=req.query[:TITLE_LEN])
        db.add(conv)
        db.commit()
        db.refresh(conv)
    else:
        conv = db.get(Conversation, req.conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="conversation not found")

    # 2) 多轮改写（无历史 / 异常时原样返回）
    rewritten = rewrite_standalone_question(db, conv.id, req.query)

    # 3) 检索（MVP 无 ACL，doc_ids 传 None；过滤已在召回期下推）
    hits = retrieve(db, rewritten, req.kb_ids, top_k=ANSWER_TOP_K)

    # 4) user message 落库（检索后、生成前）
    db.add(Message(conversation_id=conv.id, role="user", content=req.query))
    db.commit()

    logger.info(
        "chat 开始 user=%s conversation_id=%s kb_ids=%s hits=%d",
        user.user_id, conv.id, req.kb_ids, len(hits),
    )
    t0 = time.perf_counter()

    def event_stream() -> Iterator[str]:
        """SSE 生成器：delta* -> refs -> done（异常时 error 收尾并落库跳过）。"""
        parts: list[str] = []
        try:
            if not hits:
                parts.append(NO_RESULT_TEXT)
                yield _sse("delta", {"text": NO_RESULT_TEXT})
            else:
                messages = build_messages(rewritten, hits)
                stream = model_gateway.get_gateway().generate(messages, stream=True)
                if isinstance(stream, str):  # 防御：网关未按流式返回
                    stream = [stream]
                for piece in stream:
                    if not piece:
                        continue
                    parts.append(piece)
                    yield _sse("delta", {"text": piece})

            refs = _refs_payload(hits)
            yield _sse("refs", refs)

            latency_ms = int((time.perf_counter() - t0) * 1000)
            assistant = Message(
                conversation_id=conv.id,
                role="assistant",
                content="".join(parts),
                refs=refs,
                latency_ms=latency_ms,
            )
            db.add(assistant)
            db.commit()
            db.refresh(assistant)
            yield _sse(
                "done",
                {"conversation_id": conv.id, "message_id": assistant.id, "latency_ms": latency_ms},
            )
        except Exception:
            logger.exception("chat 流式生成失败 conversation_id=%s", conv.id)
            yield _sse("error", {"detail": "internal error during generation"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
