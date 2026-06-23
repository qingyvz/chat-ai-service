from typing import AsyncIterator, List, Union

from chat.application.tools import ToolScope
from chat.application.tools.core.execution.dispatcher import ToolDispatcher
from chat.application.tools.core.llm.invocation import ToolInvocation
from chat.application.tools.core.llm.renderer import tool_result_renderer
from chat.domain.entities import ChatMessage, Role
from chat.domain.entities.message import MessageModelInfo
from chat.domain.interfaces.llm import LLMEventType
from chat.domain.repositories.model_repo import ModelRequestInfo
from chat.domain.error_codes import ChatErrorCode
from chat.application.llm_provider_resolver import LLMProviderResolver
from chat.application.token_counter import TokenCounter
from common.core.exceptions import ServiceException
from chat.application.events import (
    StepFinishEvent,
    StepStartEvent,
    StreamEvent,
    ToolInputAvailableEvent,
    ToolInputStartEvent,
    ToolOutputAvailableEvent,
)
from chat.application.orchestration.delta_interpreter import StepDeltaInterpreter


class AgentStepRunner:
    """单步原语：一次 LLM turn + 工具执行，被各编排策略（ReAct / Plan-Execute Executor）共用"""

    def __init__(self, llm_provider_resolver: LLMProviderResolver, token_counter: TokenCounter) -> None:
        self._llm_provider_resolver = llm_provider_resolver
        self._token_counter = token_counter
        self._tool_dispatcher = ToolDispatcher()

    async def run(
        self,
        messages: List[ChatMessage],
        session_id: str,
        model_request: ModelRequestInfo,
        iteration: int,
        tool_scope: ToolScope,
    ) -> AsyncIterator[Union[StreamEvent, StepFinishEvent]]:
        # 发 step 开始事件
        yield StepStartEvent()

        interpreter = StepDeltaInterpreter()

        # 按 model_request 选具体 provider adapter
        llm_provider = self._llm_provider_resolver.resolve(model_request)

        # 仅在模型与 provider 均支持工具时才传 schema
        tool_schemas = tool_scope.schemas() if model_request.support_tools and llm_provider.supports_tools() else []

        token_usage = 0
        try:
            # provider 内部解析原生协议并产出 LLMStreamEvent
            async for provider_event in llm_provider.stream_chat_completion(
                messages=messages,
                model_request=model_request,
                tools=tool_schemas or None,
            ):
                if provider_event.type == LLMEventType.USAGE and provider_event.usage:
                    token_usage += provider_event.usage.total_tokens
                for event in interpreter.consume(provider_event):
                    yield event
        except ServiceException:
            raise
        except Exception as e:
            raise ServiceException(
                ChatErrorCode.LLM_GENERATION_FAILED,
                custom_msg=f"流式推理失败 (iter={iteration}): {e}",
            )

        for event in interpreter.close():
            yield event

        assistant_msg = ChatMessage(
            session_id=session_id,
            role=Role.ASSISTANT,
            model_id=model_request.model_id,
            model_info=MessageModelInfo.from_model_request(model_request),
            content=interpreter.assistant_content or "",
            reasoning_content=interpreter.assistant_reasoning or None,
            provider_payload=interpreter.provider_payload,
            tool_calls=interpreter.tool_calls or None,
        )

        if token_usage == 0:
            # 未能正确计费，兜底估算（输入 + 输出）
            token_usage += await self._token_counter.count_messages(messages=messages, model_name=model_request.model_name, tools=tool_schemas or None)
            token_usage += await self._token_counter.count_messages(messages=[assistant_msg], model_name=model_request.model_name)
        assistant_msg.token_usage = token_usage

        # 没有工具调用 → 结束本轮（也结束整个循环）
        if not interpreter.tool_calls:
            yield StepFinishEvent(is_finished=True, final_assistant_message=assistant_msg, usage_tokens=token_usage)
            return

        # 有工具调用 → 进入工具阶段
        invocations = [
            ToolInvocation(
                tool_call_id=tool_call.call_id,
                tool_name=tool_call.name,
                tool_call_arguments=tool_call.arguments,
                query_loop_iteration=iteration,
            )
            for tool_call in interpreter.tool_calls
        ]

        new_messages: List[ChatMessage] = [assistant_msg]

        for invocation in invocations:
            yield ToolInputStartEvent(call_id=invocation.tool_call_id, tool_name=invocation.tool_name)
            yield ToolInputAvailableEvent(call_id=invocation.tool_call_id, tool_name=invocation.tool_name, input=invocation.tool_call_arguments)

        tool_outputs = await self._tool_dispatcher.dispatch(invocations, tool_scope)

        for result in tool_outputs.results:
            tool = tool_scope.get(result.tool_invocation.tool_name)
            result = tool_result_renderer(result, tool.definition if tool else None)

            yield ToolOutputAvailableEvent(call_id=result.tool_call_id, output=result.tool_output)
            new_messages.append(
                ChatMessage(
                    session_id=session_id,
                    role=Role.TOOL,
                    tool_call_id=result.tool_call_id,
                    name=result.tool_name,
                    content=result.tool_output,
                    persisted_output_placeholder=result.persisted_output_placeholder,
                )
            )

        # 结束本轮并继续下一轮模型推理（因为调用工具）
        yield StepFinishEvent(is_finished=False, intermediate_messages=new_messages, usage_tokens=token_usage)
