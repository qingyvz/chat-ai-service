import uuid
from typing import Any, Dict, List

from common.logger import warn
from chat.domain.entities import Plan, PlanStep
from chat.domain.repositories import PlanRepository
from chat.service_client.resource_service_client import ResourceClient
from chat.service_client.file_storage_service_client import FileStorageClient
from chat.application.events import PlanCreatedEvent
from chat.application.orchestration.plan_context import publish_plan_content
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
    """创建 PlanMode 计划文件：注册 resource + 持久化 + 发 Kafka，待用户审查"""

    def __init__(self, plan_repo: PlanRepository, resource_client: ResourceClient, file_storage_client: FileStorageClient, kafka_producer: Any) -> None:
        self._plan_repo = plan_repo
        self._resource_client = resource_client
        self._file_storage_client = file_storage_client
        self._kafka_producer = kafka_producer
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
                required_context_keys=("plan_context", "session_id", "user_id"),
            ),
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, context: dict[str, Any], **kwargs: Any) -> str:
        plan_context = context.get("plan_context")
        session_id = context.get("session_id")
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
            session_id=str(session_id),
            user_id=str(user_id),
            file_name=file_name,
            content=content,
            steps=steps,
        )
        markdown = plan.render_markdown()

        # 直传 file-storage(OSS)，file-storage 发 file-uploaded 通知 ai-asset
        try:
            plan.object_key = await self._file_storage_client.upload_content(markdown, extension="md")
        except Exception as e:
            warn("plan content upload failed.", session_id=session_id, detail=str(e))

        # 注册为 resource（带字节大小）
        try:
            plan.resource_id = await self._resource_client.create_resource_item(
                resource_name=file_name, owner_id=str(user_id), size=len(markdown.encode("utf-8")),
            )
        except Exception as e:
            warn("plan resource register failed.", session_id=session_id, detail=str(e))

        plan.content_hash = plan.compute_hash()
        await self._plan_repo.create(plan)

        try:
            await publish_plan_content(self._kafka_producer, plan)
        except Exception as e:
            warn("plan content publish failed.", session_id=session_id, detail=str(e))

        plan_context.plan = plan
        plan_context.emit(PlanCreatedEvent(plan_id=plan.plan_id, steps=plan.steps_payload()))
        return f"[create_file] 计划「{file_name}」已创建，等待用户审查（execute / change）。"
