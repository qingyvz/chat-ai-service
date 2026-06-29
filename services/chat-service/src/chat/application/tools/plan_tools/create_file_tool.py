import uuid
from typing import Any, Dict, List

from common.core.exceptions import RpcError
from chat.domain.entities import Plan, PlanStep
from chat.core.persistence import RedisPlanCache
from chat.service_client import AIAssetClient
from chat.application.events import PlanCreatedEvent
from chat.application.tools.core import (
    ToolDefinition,
    ToolExecutionError,
    ToolLLMSpec,
    ToolParametersSchema,
    ToolPolicy,
)

_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_name": {"type": "string", "description": "计划文件名/标题"},
        "file_type": {"type": "string", "enum": ["plan"], "description": "文件类型，固定 plan"},
        "content": {"type": "string", "description": "计划正文 markdown：背景 / 任务 / 注意事项 / 思路"},
        "steps": {
            "type": "array",
            "description": "todolist 条目（按执行顺序）",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["title"],
            },
        },
    },
    "required": ["file_name", "content", "steps"],
}


class CreateFileTool:
    """创建 PlanMode 计划文件：经 ai-asset 注册资产（直传 OSS + resource）+ 热缓存，待用户审查"""

    def __init__(self, ai_asset_client: AIAssetClient, plan_cache: RedisPlanCache) -> None:
        self._ai_asset_client = ai_asset_client
        self._plan_cache = plan_cache
        self._definition = ToolDefinition(
            llm_spec=ToolLLMSpec(
                name="create_file",
                description=(
                    "Create the PLAN file for the user's request: prose content "
                    "(background / task / cautions / approach) plus a todolist of steps. "
                    "The plan is saved as a reviewable file; the user then approves (execute) or requests changes."
                ),
                parameters_schema=ToolParametersSchema(_PARAMS),
            ),
            policy=ToolPolicy(
                expose_by_default=True,
                persist_output=False,
                required_context_keys=("plan_context", "user_id"),
            ),
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, context: dict[str, Any], **kwargs: Any) -> str:
        plan_context = context.get("plan_context")
        user_id = context.get("user_id")
        if plan_context is None:
            raise ToolExecutionError(reason="Plan Context Missing", detail_reason="create_file 只能在 PlanMode 编排内调用。")

        file_name = (kwargs.get("file_name") or "计划").strip()
        content = kwargs.get("content") or ""
        raw_steps: List[dict] = kwargs.get("steps") or []
        steps = [
            PlanStep(
                step_id=f"step_{i + 1}_{uuid.uuid4().hex[:6]}",
                title=(s.get("title") or "").strip(),
                description=(s.get("description") or "").strip(),
            )
            for i, s in enumerate(raw_steps) if (s.get("title") or "").strip()
        ]

        plan = Plan(
            plan_id=f"plan_{uuid.uuid4().hex}",
            user_id=str(user_id),
            file_name=file_name,
            content=content,
            steps=steps,
        )

        try:
            resp = await self._ai_asset_client.create_plan(
                owner_id=str(user_id), title=file_name,
                content=plan.render_markdown(), steps=plan.steps_payload(),
            )
        except RpcError as e:
            raise ToolExecutionError(reason="Plan Register Failed", detail_reason=f"计划资产注册失败：{e}")

        plan.resource_id = resp.get("resourceId")
        plan.object_key = resp.get("objectKey")
        plan.content_hash = plan.compute_hash()
        await self._plan_cache.save(plan.user_id, plan)

        plan_context.plan = plan
        plan_context.emit(PlanCreatedEvent(plan_id=plan.plan_id, steps=plan.steps_payload()))
        return f"[create_file] 计划「{file_name}」已创建，等待用户审查（execute / change）。"
