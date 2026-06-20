from typing import Dict, Iterator

from chat.application.tools.core.llm.invocation import ToolCallMessageAccumulator
from chat.application.events import (
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
)


class StepDeltaInterpreter:
    """单个 Agent Step 内的 Delta 解释器：按到达顺序消费 LLM delta，维护 reasoning/text 生命周期并产出 StreamEvent"""
    def __init__(self, text_id: str, reasoning_id: str) -> None:
        self.text_id = text_id
        self.reasoning_id = reasoning_id
        self.assistant_content: str = ""
        self.assistant_reasoning: str = ""
        self.tool_call_message_accumulators: Dict[int, ToolCallMessageAccumulator] = {}
        self._text_started: bool = False
        self._reasoning_started: bool = False

    def consume(self, delta) -> Iterator[StreamEvent]:
        """按到达顺序消费 LLM 的 delta 片段，并产出 0..N 个 StreamEvent"""
        # 若 delta.reasoning_content 有值
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            # 若 reasoning 还没开始，发 ReasoningStartEvent
            if not self._reasoning_started:
                yield ReasoningStartEvent(reasoning_id=self.reasoning_id)
                self._reasoning_started = True
            # 把 reasoning 累加到 assistant_reasoning
            self.assistant_reasoning += delta.reasoning_content
            # 发 ReasoningDeltaEvent
            yield ReasoningDeltaEvent(
                reasoning_id=self.reasoning_id,
                delta=delta.reasoning_content,
            )
        # 若 delta.content 有值
        if delta.content:
            # 若文本流还没开始
            if not self._text_started:
                # 若 reasoning 未结束，发 ReasoningEndEvent
                if self._reasoning_started:
                    yield ReasoningEndEvent(reasoning_id=self.reasoning_id)
                    self._reasoning_started = False
                # 发 TextStartEvent
                yield TextStartEvent(text_id=self.text_id)
                self._text_started = True
            # 把文本累加到 assistant_content
            self.assistant_content += delta.content
            # 发 TextDeltaEvent
            yield TextDeltaEvent(
                text_id=self.text_id,
                delta=delta.content,
            )
        # 若 delta.tool_calls 有值
        if delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                # 按 index 找到对应 accumulator
                idx = tool_call_delta.index
                if idx not in self.tool_call_message_accumulators:
                    self.tool_call_message_accumulators[idx] = ToolCallMessageAccumulator()
                if tool_call_delta.id: # 累加 id（如果有）
                    self.tool_call_message_accumulators[idx].tool_call_id = tool_call_delta.id
                if tool_call_delta.function: # 累加 function（如果有）
                    if tool_call_delta.function.name: # 累加 name
                        self.tool_call_message_accumulators[idx].tool_name += tool_call_delta.function.name
                    if tool_call_delta.function.arguments: # 累加 arguments
                        self.tool_call_message_accumulators[idx].tool_call_argument_str += tool_call_delta.function.arguments
        # tool_call 只有在一整轮模型输出结束后，才能确定是不是完整、能不能解析

    def close(self) -> Iterator[StreamEvent]:
        """在模型流结束后补齐未闭合的 reasoning/text 生命周期，该方法应在单轮 stream 结束后调用一次"""
        if self._reasoning_started:
            yield ReasoningEndEvent(reasoning_id=self.reasoning_id)
            self._reasoning_started = False
        if self._text_started:
            yield TextEndEvent(text_id=self.text_id)
            self._text_started = False
