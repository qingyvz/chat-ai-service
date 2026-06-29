import uuid
from typing import Any, Dict

from chat.core.config.app_settings import settings
from chat.application.events import StepFinishEvent
from chat.application.runtime.agent_turn_runtime import AgentTurnRuntime
from chat.application.runtime.agent_provider import StaticAgentProvider, build_subagent_info
from chat.application.runtime.context_provider import IsolatedContextProvider
from chat.application.runtime.model_resolver import InheritedModelResolver
from chat.application.tools.core import (
    ToolDefinition,
    ToolExecutionError,
    ToolLLMSpec,
    ToolParametersSchema,
    ToolPolicy,
    ToolRiskLevel,
)


class CallSubAgentTool:
    """把自包含子任务交给一个隔离 subagent 跑到结束，只返回其结论文本；从 context 取共享服务当场组装子轮直调 handle_chat，不落库"""

    def __init__(self) -> None:
        parameters_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The self-contained subtask for the sub-agent to complete, in the user's language.",
                },
                "role": {
                    "type": "string",
                    "description": "Optional short role name for the sub-agent, e.g. 'executor' or 'researcher'.",
                },
            },
            "required": ["task"],
        }
        self._definition = ToolDefinition(
            llm_spec=ToolLLMSpec(
                name="call_subagent",
                description=(
                    "Delegate a self-contained subtask to an isolated sub-agent. "
                    "Runs it to completion and returns only its result text."
                ),
                parameters_schema=ToolParametersSchema(parameters_schema),
            ),
            policy=ToolPolicy(
                expose_by_default=True,
                persist_output=True,
                risk_level=ToolRiskLevel.MEDIUM,
                required_context_keys=("session_id", "user_id", "runtime_services", "parent_model", "parent_agent_spec"),
                max_output_chars=settings.TOOL_RESULT_MAX_CHARS,
            ),
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> str:
        services = context.get("runtime_services")
        parent_model = context.get("parent_model")
        parent_spec = context.get("parent_agent_spec")
        session_id = context.get("session_id")
        user_id = context.get("user_id")
        if services is None or parent_model is None or parent_spec is None or not session_id or not user_id:
            raise ToolExecutionError(
                reason="subagent_unavailable",
                detail_reason="Sub-agent calling context is missing (no runtime services/model/spec).",
            )

        task = (kwargs.get("task") or "").strip()
        if not task:
            raise ToolExecutionError(
                reason="missing_argument",
                detail_reason="task is required.",
            )
        role = (kwargs.get("role") or "executor").strip() or "executor"

        # 当场造收窄 spec，复用共享服务 + 换上 SubAgent* 三件组装子轮，直调其 handle_chat 跑到结束
        agent_info = build_subagent_info(parent_spec, role, f"subagent_{uuid.uuid4().hex[:8]}")
        sub_runtime = AgentTurnRuntime(
            services=services,
            agent_provider=StaticAgentProvider(agent_info),
            model_resolver=InheritedModelResolver(parent_model),
            context_provider=IsolatedContextProvider(),
        )
        final_text = ""
        async for event in sub_runtime.handle_chat(
            user_id=user_id, session_id=session_id, user_query=task, background_tasks=None,
        ):
            if isinstance(event, StepFinishEvent) and event.is_finished and event.final_assistant_message is not None:
                final_text = event.final_assistant_message.content or final_text
        return final_text
