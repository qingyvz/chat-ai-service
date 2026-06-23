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


class ExecuteStepTool:
    """派发一个计划步到隔离 executor 执行并返回结论；并行来自 model 一轮同发多个本工具调用（复用 ToolDispatcher 的 gather）"""

    def __init__(self) -> None:
        parameters_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "step_id": {
                    "type": "string",
                    "description": "The id of a plan step whose dependencies are all completed.",
                },
            },
            "required": ["step_id"],
        }
        self._definition = ToolDefinition(
            llm_spec=ToolLLMSpec(
                name="execute_step",
                description=(
                    "Execute one ready plan step (all its dependencies completed) in isolation and return its result. "
                    "Call it for several independent ready steps in the SAME turn to run them in parallel."
                ),
                parameters_schema=ToolParametersSchema(parameters_schema),
            ),
            policy=ToolPolicy(
                expose_by_default=True,
                persist_output=True,
                risk_level=ToolRiskLevel.MEDIUM,
                required_context_keys=("plan_session",),
                max_output_chars=settings.TOOL_RESULT_MAX_CHARS,
            ),
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> str:
        session = context.get("plan_session")
        if session is None or session.executor is None:
            raise ToolExecutionError(
                reason="plan_unavailable",
                detail_reason="execute_step is only available inside a Plan-and-Execute turn.",
            )

        step_id = (kwargs.get("step_id") or "").strip()
        step = session.get_step(step_id)
        if step is None:
            return f"[execute_step] 未知的 step_id: {step_id}"
        if not session.is_ready(step):
            return f"[execute_step] step {step_id} 未就绪（依赖未完成或已终止），请先完成依赖或重规划。"

        session.start(step_id)
        dep_results = session.dep_results(step)
        try:
            async with session.semaphore:
                result = await session.executor.run(step.title, step.description, dep_results)
        except Exception as e:  # 内层执行异常 → 记失败，交自动机重规划
            session.step_tokens += 0
            session.mark_failed(step_id, f"{type(e).__name__}: {e}")
            return f"[execute_step] step {step_id} 执行异常：{type(e).__name__}: {e}"

        session.step_tokens += result.tokens
        if result.ok:
            session.mark_completed(step_id, result.text)
            return result.text
        session.mark_failed(step_id, result.text)
        return f"[execute_step] step {step_id} 执行失败：{result.text}"
