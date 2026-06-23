from typing import Any, Dict

from chat.application.tools.core import (
    ToolDefinition,
    ToolExecutionError,
    ToolLLMSpec,
    ToolParametersSchema,
    ToolPolicy,
    ToolRiskLevel,
)


class UpdatePlanTool:
    """重规划：按传入步重建计划（保留已完成步结果），用于失败修复或调整 DAG；推 PlanUpdatedEvent 覆盖渲染"""

    def __init__(self) -> None:
        parameters_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "The full revised step list (replaces the current plan).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Existing step id to keep; omit for a new step."},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "depends_on": {"type": "array", "items": {"type": "string"}, "description": "step_ids this step depends on."},
                        },
                        "required": ["title"],
                    },
                },
            },
            "required": ["steps"],
        }
        self._definition = ToolDefinition(
            llm_spec=ToolLLMSpec(
                name="update_plan",
                description=(
                    "Revise the plan to fix a failed step or adjust the DAG. Pass the full revised step list; "
                    "completed steps (matched by step_id) keep their results, others become pending."
                ),
                parameters_schema=ToolParametersSchema(parameters_schema),
            ),
            policy=ToolPolicy(
                expose_by_default=True,
                persist_output=False,
                risk_level=ToolRiskLevel.LOW,
                required_context_keys=("plan_session",),
            ),
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> str:
        session = context.get("plan_session")
        if session is None:
            raise ToolExecutionError(
                reason="plan_unavailable",
                detail_reason="update_plan is only available inside a Plan-and-Execute turn.",
            )
        steps = kwargs.get("steps") or []
        if not isinstance(steps, list) or not steps:
            return "[update_plan] steps 不能为空。"
        session.replace_steps(steps)
        return "[update_plan] 计划已更新。"
