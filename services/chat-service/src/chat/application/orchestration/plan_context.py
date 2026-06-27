from typing import List, Optional

from chat.domain.entities import Plan
from chat.application.events import StreamEvent


class PlanContext:
    """PlanMode turn 级共享状态：工具写当前 plan + pending 事件，策略 drain 后 yield"""

    def __init__(self) -> None:
        self.plan: Optional[Plan] = None
        self._events: List[StreamEvent] = []

    def emit(self, event: StreamEvent) -> None:
        self._events.append(event)

    def drain_events(self) -> List[StreamEvent]:
        events = self._events
        self._events = []
        return events
