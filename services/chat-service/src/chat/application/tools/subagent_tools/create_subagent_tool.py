from typing import Any, Dict

from chat.application.tools.core import (
    ToolDefinition,
    ToolExecutionError,
    ToolLLMSpec,
    ToolParametersSchema,
    ToolPolicy,
    ToolRiskLevel,
)


class CreateSubAgentTool:
    """创建一个收窄的 subagent（落 Redis）并返回 subagent_id；机制由 context 注入的 subagent_spawner 承载，工具本身无状态"""

    def __init__(self) -> None:
        parameters_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "description": "Short role name for the sub-agent, e.g. 'executor' or 'researcher'.",
                },
            },
            "required": [],
        }
        self._definition = ToolDefinition(
            llm_spec=ToolLLMSpec(
                name="create_subagent",
                description=(
                    "Create an isolated sub-agent to delegate a focused subtask to. "
                    "Returns a subagent_id; pass it to call_subagent to run the subtask."
                ),
                parameters_schema=ToolParametersSchema(parameters_schema),
            ),
            policy=ToolPolicy(
                expose_by_default=True,
                persist_output=False,
                risk_level=ToolRiskLevel.LOW,
                required_context_keys=("session_id", "subagent_spawner", "parent_agent_spec"),
            ),
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> str:
        spawner = context.get("subagent_spawner")
        parent_spec = context.get("parent_agent_spec")
        session_id = context.get("session_id")
        if spawner is None or parent_spec is None or not session_id:
            raise ToolExecutionError(
                reason="subagent_unavailable",
                detail_reason="Sub-agent spawning context is missing (no spawner/parent spec).",
            )
        role = (kwargs.get("role") or "executor").strip() or "executor"
        return await spawner.create(session_id=session_id, parent_spec=parent_spec, role=role)
