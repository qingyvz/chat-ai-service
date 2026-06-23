from typing import Any, Dict

from chat.core.config.app_settings import settings
from chat.application.tools.core import (
    ToolDefinition,
    ToolExecutionError,
    ToolLLMSpec,
    ToolParametersSchema,
    ToolPolicy,
    ToolRiskLevel,
)


class CallSubAgentTool:
    """把子任务交给指定 subagent 隔离运行到结束，只返回其结论文本；机制由 context 注入的 subagent_spawner 承载，工具本身无状态"""

    def __init__(self) -> None:
        parameters_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "subagent_id": {
                    "type": "string",
                    "description": "The sub-agent id returned by create_subagent.",
                },
                "task": {
                    "type": "string",
                    "description": "The self-contained subtask for the sub-agent to complete, in the user's language.",
                },
            },
            "required": ["subagent_id", "task"],
        }
        self._definition = ToolDefinition(
            llm_spec=ToolLLMSpec(
                name="call_subagent",
                description=(
                    "Delegate a self-contained subtask to a sub-agent created via create_subagent. "
                    "Runs it in isolation and returns only its result text."
                ),
                parameters_schema=ToolParametersSchema(parameters_schema),
            ),
            policy=ToolPolicy(
                expose_by_default=True,
                persist_output=True,
                risk_level=ToolRiskLevel.MEDIUM,
                required_context_keys=("session_id", "user_id", "subagent_spawner", "parent_model"),
                max_output_chars=settings.TOOL_RESULT_MAX_CHARS,
            ),
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> str:
        spawner = context.get("subagent_spawner")
        parent_model = context.get("parent_model")
        session_id = context.get("session_id")
        user_id = context.get("user_id")
        if spawner is None or parent_model is None or not session_id or not user_id:
            raise ToolExecutionError(
                reason="subagent_unavailable",
                detail_reason="Sub-agent calling context is missing (no spawner/parent model).",
            )

        subagent_id = (kwargs.get("subagent_id") or "").strip()
        task = (kwargs.get("task") or "").strip()
        if not subagent_id or not task:
            raise ToolExecutionError(
                reason="missing_argument",
                detail_reason="Both subagent_id and task are required.",
            )

        # prior_results 仅供策略直调时透传前序步骤结论（LLM 调用时缺省为空）
        prior_results = kwargs.get("prior_results") or []
        return await spawner.call(
            session_id=session_id,
            user_id=user_id,
            subagent_id=subagent_id,
            parent_model=parent_model,
            task=task,
            prior_results=prior_results,
        )
