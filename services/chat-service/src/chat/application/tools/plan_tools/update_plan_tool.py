from typing import Any, Dict

from common.logger import warn
from chat.domain.repositories import PlanRepository
from chat.application.events import PlanStepStatusEvent, PlanUpdatedEvent
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
        "step_id": {"type": "string", "description": "要更新状态的 todolist 步骤 id"},
        "status": {
            "type": "string",
            "enum": ["pending", "in_progress", "completed", "failed"],
            "description": "步骤新状态",
        },
        "result_summary": {"type": "string", "description": "该步骤的简短结果（可选）"},
        "content": {"type": "string", "description": "重写计划正文 markdown（change / 重规划时用，可选）"},
    },
}


class UpdatePlanTool:
    """更新 PlanMode 计划：翻转 todolist 步骤状态或重写正文，持久化 + 发 Kafka + 上报事件"""

    def __init__(self, plan_repo: PlanRepository, kafka_producer: Any) -> None:
        self._plan_repo = plan_repo
        self._kafka_producer = kafka_producer
        self._definition = ToolDefinition(
            llm_spec=ToolLLMSpec(
                name="update_plan",
                description=(
                    "Update the active plan: set a todolist step's status (in_progress / completed / failed) "
                    "as you execute it, or rewrite the plan content when revising."
                ),
                parameters_schema=ToolParametersSchema(_PARAMS),
            ),
            policy=ToolPolicy(
                expose_by_default=True,
                persist_output=False,
                required_context_keys=("plan_context",),
            ),
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, context: dict[str, Any], **kwargs: Any) -> str:
        plan_context = context.get("plan_context")
        if plan_context is None or plan_context.plan is None:
            raise ToolExecutionError(reason="No Active Plan", detail_reason="update_plan 需要一个进行中的计划。")
        plan = plan_context.plan

        content = kwargs.get("content")
        step_id = kwargs.get("step_id")
        status = kwargs.get("status")

        if content is not None:
            plan.content = content

        step_event = None
        if step_id and status:
            step = next((s for s in plan.steps if s.step_id == step_id), None)
            if step is None:
                return f"[update_plan] 未知 step_id: {step_id}"
            step.status = status
            if kwargs.get("result_summary"):
                step.result_summary = kwargs["result_summary"]
            step_event = PlanStepStatusEvent(step_id=step_id, status=status, result_summary=step.result_summary)

        plan.content_hash = plan.compute_hash()
        await self._plan_repo.save(plan)
        try:
            await publish_plan_content(self._kafka_producer, plan)
        except Exception as e:
            warn("plan content publish failed.", detail=str(e))

        if step_event is not None:
            plan_context.emit(step_event)
        if content is not None:
            plan_context.emit(PlanUpdatedEvent(plan_id=plan.plan_id, steps=plan.steps_payload()))
        return "[update_plan] 计划已更新。"
