import uuid
from typing import List, Optional, AsyncIterator

from beanie import PydanticObjectId

from chat.application.tools import ToolScope
from common.logger import warn
from chat.core.config.app_settings import settings
from chat.domain.entities import ChatMessage, Role
from chat.domain.interfaces import LLMProvider
from chat.application.orchestration import AgentStepRunner
from chat.application.events import (
    StepFinishEvent,
    StepStartEvent,
    StreamEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
)


class QueryLoopRuntime:
    """ReAct 多轮循环：委派单步给 AgentStepRunner，按 StepFinishEvent 决定是否继续下一轮（while + MAX_ITERATIONS）"""

    def __init__(self, llm: LLMProvider) -> None:
        self._step_runner = AgentStepRunner(llm)

    async def stream_chat_with_tool_calling(
        self,
        messages: List[ChatMessage],
        tool_scope: ToolScope,
        session_id: str,
        agent_max_iterations: Optional[int],
        model_name: str,
        model_id: Optional[PydanticObjectId] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        # 进入多轮循环
        for iteration in range(agent_max_iterations or settings.AGENT_MAX_ITERATIONS):
            step_finish_event: Optional[StepFinishEvent] = None
            # 把当前 messages、模型参数 和 tool_scope 委派给单步原语，然后异步消费它的产出
            async for item in self._step_runner.run(
                messages=messages,
                session_id=session_id,
                model_name=model_name,
                model_id=model_id,
                api_base=api_base,
                api_key=api_key,
                iteration=iteration,
                tool_scope=tool_scope,
            ):
                # 如果拿到的是 StepFinishEvent 就存到 step_finish_event；否则直接 yield
                if isinstance(item, StepFinishEvent):
                    step_finish_event = item
                yield item

            assert step_finish_event is not None
            if step_finish_event.is_finished:
                return
            else:
                # 统一追加消息并决定是否继续下一轮
                messages.extend(step_finish_event.intermediate_messages)
        else:
            # 超出最大迭代次数时兜底
            async for event in self._emit_exhausted_warning(session_id):
                yield event

    async def _emit_exhausted_warning(
        self, session_id: str
    ) -> AsyncIterator[StreamEvent]:
        """Agent 循环超出最大迭代次数时的兜底文本输出"""
        warning_text = f"Agent 推理超出最大迭代次数{settings.AGENT_MAX_ITERATIONS}，未能生成最终答案"
        warn("tool calling loop exhausted.", session_id=session_id)
        text_id = f"txt_{uuid.uuid4().hex}"
        yield StepStartEvent()
        yield TextStartEvent(text_id=text_id)
        yield TextDeltaEvent(text_id=text_id, delta=warning_text)
        yield TextEndEvent(text_id=text_id)
        final_message = ChatMessage(
            session_id=session_id,
            role=Role.ASSISTANT,
            content=warning_text,
        )
        yield StepFinishEvent(is_finished=True, final_assistant_message=final_message, usage_tokens=0)
