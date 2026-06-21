import uuid
from typing import Iterator, Optional

from chat.domain.entities.message import ToolCallMessage
from chat.domain.interfaces.llm import LLMEventType, LLMStreamEvent
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
    """单个 Agent Step 内的事件解释器：按到达顺序消费 LLMStreamEvent，维护 reasoning/text 生命周期、收集 tool_calls/provider_payload，产出 StreamEvent"""

    def __init__(self) -> None:
        self.text_id = f"txt_{uuid.uuid4().hex}"
        self.reasoning_id = f"rsn_{uuid.uuid4().hex}"
        self.assistant_content: str = ""
        self.assistant_reasoning: str = ""
        self.tool_calls: list[ToolCallMessage] = []
        self.provider_payload: Optional[dict] = None
        self._text_started: bool = False
        self._reasoning_started: bool = False

    def consume(self, item: LLMStreamEvent) -> Iterator[StreamEvent]:
        """按到达顺序消费 LLMStreamEvent，产出 0..N 个 StreamEvent（不处理 USAGE，由外层累加）"""
        # 原生载荷
        if item.type == LLMEventType.STATE:
            self.provider_payload = item.provider_payload
            return
        # 工具调用列表（整轮输出结束后才进入工具执行阶段）
        if item.type == LLMEventType.TOOL_CALLS:
            self.tool_calls.extend(item.tool_calls or [])
            return
        # reasoning 增量
        if item.type == LLMEventType.REASONING_DELTA and item.delta:
            if not self._reasoning_started:
                yield ReasoningStartEvent(reasoning_id=self.reasoning_id)
                self._reasoning_started = True
            self.assistant_reasoning += item.delta
            yield ReasoningDeltaEvent(reasoning_id=self.reasoning_id, delta=item.delta)
            return
        # 文本增量
        if item.type == LLMEventType.TEXT_DELTA and item.delta:
            if not self._text_started:
                if self._reasoning_started:
                    yield ReasoningEndEvent(reasoning_id=self.reasoning_id)
                    self._reasoning_started = False
                yield TextStartEvent(text_id=self.text_id)
                self._text_started = True
            self.assistant_content += item.delta
            yield TextDeltaEvent(text_id=self.text_id, delta=item.delta)

    def close(self) -> Iterator[StreamEvent]:
        """模型流结束后补齐未闭合的 reasoning/text 生命周期，单轮 stream 结束后调用一次"""
        if self._reasoning_started:
            yield ReasoningEndEvent(reasoning_id=self.reasoning_id)
            self._reasoning_started = False
        if self._text_started:
            yield TextEndEvent(text_id=self.text_id)
            self._text_started = False
