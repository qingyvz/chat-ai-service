import asyncio
import uuid

from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from dependency_injector.wiring import inject, Provide

from chat.api.vercel_formats import (
    message_start, message_finish, stream_done, abort, error as sse_error,
)
from chat.api.vercel_sse_mapper import to_vercel_sse

from common.security import require_login
from common.logger import error, info
from chat.api.schemas.chat import ChatRequest
from chat.application.runtime import AgentTurnRuntime
from chat.container import Container
from chat.core.config.app_settings import settings
from chat.domain.repositories import SessionRepository

router = APIRouter()


async def _vercel_generator(chat_gen, model_name: str):
    """将 coordinator 的 AsyncGenerator 包装成 AI SDK 6.x SSE 格式"""
    message_id = f"msg_{uuid.uuid4().hex}"
    try:
        yield message_start(message_id)

        async for event in chat_gen:
            yield to_vercel_sse(event)

        yield message_finish()
        yield stream_done()

    except asyncio.CancelledError:
        info("chat stream generation cancelled.")
        yield abort(reason="user_cancelled")
        yield stream_done()
        raise

    except Exception as e:
        error("chat stream generation failed.", exc=e)
        yield sse_error(error_text=str(e))
        yield stream_done()


async def _stream_chat(
        req: ChatRequest,
        background_tasks: BackgroundTasks,
        user_id: str,
        coordinator: AgentTurnRuntime,
        session_repo: SessionRepository,
        think_type_override: str | None = None,
) -> StreamingResponse:
    """两个 endpoint 共用：校验 → 调 handle_chat（差别仅 think_type_override）→ 包成 Vercel SSE"""
    if not req.query:
        raise HTTPException(status_code=400, detail="缺少查询内容")

    if not req.session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")

    resolved_model_id = PydanticObjectId(req.model or settings.DEFAULT_MODEL_ID)
    resolved_provider_id = PydanticObjectId(req.provider_id) if req.provider_id else None

    await session_repo.get_session_for_user(req.session_id, user_id)

    chat_gen = coordinator.handle_chat(
        user_id=user_id,
        session_id=req.session_id,
        user_query=req.query,
        background_tasks=background_tasks,
        model_id=resolved_model_id,
        provider_id=resolved_provider_id,
        frontend_states=req.frontend_states,
        user_defined_allow_tool_names=req.user_defined_allow_tool_names,
        user_defined_deny_tool_names=req.user_defined_deny_tool_names,
        user_defined_on_demand_skill_ids=req.user_defined_on_demand_skill_ids,
        user_defined_force_enabled_skill_ids=req.user_defined_force_enabled_skill_ids,
        think_type_override=think_type_override,
        plan_review_decision=req.plan_review_decision,
    )

    return StreamingResponse(
        _vercel_generator(chat_gen, str(resolved_model_id)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "x-vercel-ai-ui-message-stream": "v1",
        },
    )


@router.post("/completions")
@inject
async def chat_completions(
        req: ChatRequest,
        background_tasks: BackgroundTasks,
        user_id: str = Depends(require_login),
        coordinator: AgentTurnRuntime = Depends(Provide[Container.agent_turn_runtime]),
        session_repo: SessionRepository = Depends(Provide[Container.session_repo]),
):
    """
    请求格式:
       {
         "session_id": "xxx",
         "query": "你好",
         "model": "Mongo ObjectId string",
         "provider_id": "Mongo ObjectId string",
         "states": [{
            "key": "selected_text",
            "value": "xxx",
            "disabled": false}
         ]
       }
    """
    return await _stream_chat(req, background_tasks, user_id, coordinator, session_repo)


@router.post("/completions/plan-mode")
@inject
async def chat_completions_plan_mode(
        req: ChatRequest,
        background_tasks: BackgroundTasks,
        user_id: str = Depends(require_login),
        coordinator: AgentTurnRuntime = Depends(Provide[Container.agent_turn_runtime]),
        session_repo: SessionRepository = Depends(Provide[Container.session_repo]),
):
    """PlanMode 编排入口：DTO 与 /completions 一致（含 plan_review_decision），强制 think_type=PlanMode"""
    return await _stream_chat(req, background_tasks, user_id, coordinator, session_repo, think_type_override="PlanMode")

