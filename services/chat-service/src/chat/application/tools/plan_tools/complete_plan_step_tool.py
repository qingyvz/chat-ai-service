from typing import Any, Dict

from chat.application.tools.core import (
    ToolDefinition,
    ToolExecutionError,
    ToolLLMSpec,
    ToolParametersSchema,
    ToolPolicy,
    ToolRiskLevel,
)


class CompletePlanStepTool:
    """Plan-Execute 进度上报：model 做完一步时调用，编排据此发 PlanStepStatusEvent；工具本身只回 ack"""

    def __init__(self) -> None:
        parameters_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "step_id": {
                    "type": "string",
                    "description": "The id of the plan step that has just been completed.",
                },
                "summary": {
                    "type": "string",
                    "description": "A short summary of the step's result, in the user's language.",
                },
            },
            "required": ["step_id"],
        }
        self._definition = ToolDefinition(
            llm_spec=ToolLLMSpec(
                name="complete_plan_step",
                description=(
                    "Report that a plan step is finished. Call this right after completing each step "
                    "so progress can be shown to the user. Pass the step_id from the plan and a short summary."
                ),
                parameters_schema=ToolParametersSchema(parameters_schema),
            ),
            policy=ToolPolicy(
                expose_by_default=True,
                persist_output=False,
                risk_level=ToolRiskLevel.LOW,
            ),
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> str:
        step_id = (kwargs.get("step_id") or "").strip()
        if not step_id:
            raise ToolExecutionError(
                reason="missing_argument",
                detail_reason="step_id is required.",
            )
        return f"Recorded completion of plan step {step_id}."
