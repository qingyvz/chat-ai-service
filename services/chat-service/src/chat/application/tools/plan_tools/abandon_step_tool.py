from typing import Any, Dict

from chat.application.tools.core import (
    ToolDefinition,
    ToolExecutionError,
    ToolLLMSpec,
    ToolParametersSchema,
    ToolPolicy,
    ToolRiskLevel,
)


class AbandonStepTool:
    """显式放弃一个无法完成的步：标终态失败并剪掉其下游，让其余可达步继续（"完成能完成的部分"的出口）"""

    def __init__(self) -> None:
        parameters_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "step_id": {"type": "string", "description": "The id of the step to give up on."},
                "reason": {"type": "string", "description": "Short reason why it cannot be completed."},
            },
            "required": ["step_id"],
        }
        self._definition = ToolDefinition(
            llm_spec=ToolLLMSpec(
                name="abandon_step",
                description=(
                    "Give up on a step that cannot be completed even after re-planning. "
                    "Marks it failed and skips its dependents; independent steps continue."
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
                detail_reason="abandon_step is only available inside a Plan-and-Execute turn.",
            )
        step_id = (kwargs.get("step_id") or "").strip()
        if session.get_step(step_id) is None:
            return f"[abandon_step] 未知的 step_id: {step_id}"
        reason = (kwargs.get("reason") or "model 判定该步无法完成").strip()
        session.abandon(step_id, reason)
        return f"[abandon_step] 已放弃 step {step_id} 并跳过其下游。"
