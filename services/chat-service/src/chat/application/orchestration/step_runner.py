import json
import uuid
from typing import AsyncIterator, Optional, List, Union

from beanie import PydanticObjectId

from chat.application.tools import ToolScope
from chat.application.tools.core.execution.dispatcher import ToolDispatcher
from chat.application.tools.core.llm.invocation import tool_call_parse
from chat.application.tools.core.llm.renderer import tool_result_renderer
from chat.core.config.app_settings import settings
from chat.domain.entities import ChatMessage, Role
from chat.domain.interfaces import LLMProvider
from chat.domain.error_codes import ChatErrorCode
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

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm
        self._tool_dispatcher = ToolDispatcher()

    async def run(
        self,
        messages: List[ChatMessage],
        session_id: str,
        model_name: str,
        model_id: Optional[PydanticObjectId],
        api_base: Optional[str],
        api_key: Optional[str],
        iteration: int,
        tool_scope: ToolScope,
    ) -> AsyncIterator[Union[StreamEvent, StepFinishEvent]]:
        # 发 step 开始事件
        yield StepStartEvent()

        # 创建本轮推理的 delta 解释器
        text_id = f"txt_{uuid.uuid4().hex}"
        reasoning_id = f"rsn_{uuid.uuid4().hex}"
        delta_interpreter = StepDeltaInterpreter(text_id=text_id, reasoning_id=reasoning_id)

        finish_reason: str = "stop"

        # schema 已由 ToolScope 在构造期固化，这里直读
        tool_schemas = tool_scope.schemas()

        usage_tokens = 0
        try:
            # 调用模型流式接口
            async for chunk in self.llm.stream_chat_completion(
                messages=messages,
                model_name=model_name,
                tools=tool_schemas or None,
                api_base=api_base,
                api_key=api_key,
            ):
                usage_tokens += chunk.usage_tokens
                choices = chunk.raw.choices
                if choices: # usage chunk 的 choices 可能是空数组
                    finish_reason = choices[0].finish_reason or finish_reason

                    # 把 delta 片段交给解释器，产出 StreamEvent
                    for event in delta_interpreter.consume(choices[0].delta):
                        yield event
        except ServiceException:
            raise  # 已经是业务异常，直接向上传播
        except Exception as e:
            raise ServiceException(
                ChatErrorCode.LLM_GENERATION_FAILED,
                custom_msg=f"流式推理失败 (iter={iteration}): {e}",
            )

        # 关闭本轮推理的 delta 解释器
        for event in delta_interpreter.close():
            yield event

        if usage_tokens == 0:
            # 未能正确计费，需要兜底
            usage_tokens = await self.llm.count_message_tokens(messages=messages, model_name=model_name, tools=tool_schemas or None)
            output_text = delta_interpreter.assistant_content + delta_interpreter.assistant_reasoning
            for idx in delta_interpreter.tool_call_message_accumulators.keys():
                acc = delta_interpreter.tool_call_message_accumulators[idx]
                output_text += acc.tool_call_id + acc.tool_name + acc.tool_call_argument_str
            usage_tokens += await self.llm.count_tokens(text=output_text, model_name=model_name)

        # 如果没有工具调用，则结束这一轮（也结束整个循环）
        if finish_reason != "tool_calls" or not delta_interpreter.tool_call_message_accumulators:
            final_message = ChatMessage(
                session_id=session_id,
                role=Role.ASSISTANT,
                model_id=model_id,
                content=delta_interpreter.assistant_content or "",
                reasoning_content=delta_interpreter.assistant_reasoning or None,
            )
            yield StepFinishEvent(is_finished=True, final_assistant_message=final_message, usage_tokens=usage_tokens)
            return

        # 如果有工具调用，则进入工具阶段

        # 解析工具调用
        invocations = tool_call_parse(
            delta_interpreter.tool_call_message_accumulators,
            query_loop_iteration=iteration,
        )

        # 构造 assistant 的 tool_calls 消息(OpenAI 协议要求)
        # 放入 new_messages,由外层编排策略统一 extend 进 messages
        assistant_msg = ChatMessage(
            session_id=session_id,
            role=Role.ASSISTANT,
            model_id=model_id,
            content=delta_interpreter.assistant_content or None,
            reasoning_content=delta_interpreter.assistant_reasoning or None,
            tool_calls=[
                {
                    "id": invocation.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": invocation.tool_name,
                        "arguments": json.dumps(invocation.tool_call_arguments),
                    },
                }
                for invocation in invocations
            ],
        )
        new_messages: List[ChatMessage] = [assistant_msg]

        for invocation in invocations:
            # 为每个 parsed tool_call 产生两阶段 input 事件（start + available）
            yield ToolInputStartEvent(
                call_id=invocation.tool_call_id,
                tool_name=invocation.tool_name,
            )
            yield ToolInputAvailableEvent(
                call_id=invocation.tool_call_id,
                tool_name=invocation.tool_name,
                input=invocation.tool_call_arguments,
            )

        # 通过工具 core 并发执行并归约结果
        tool_outputs = await self._tool_dispatcher.dispatch(invocations, tool_scope)

        for result in tool_outputs.results:
            tool = tool_scope.get(result.tool_invocation.tool_name)
            result = tool_result_renderer(result, tool.definition if tool else None)

            yield ToolOutputAvailableEvent(
                call_id=result.tool_call_id,
                output=result.tool_output,
            )
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
        yield StepFinishEvent(is_finished=False, intermediate_messages=new_messages, usage_tokens=usage_tokens)
