from typing import List, Optional

from chat.domain.entities import Plan
from chat.application.events import StreamEvent
from chat.core.config.app_settings import settings


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


async def publish_plan_content(kafka_producer, plan: Plan) -> None:
    """把计划全文快照发到 plan-content-topic，供 resource 入全文检索（无 resource_id 跳过）"""
    if not plan.resource_id:
        return
    markdown = plan.render_markdown()
    await kafka_producer.send(
        topic=settings.KAFKA_PLAN_CONTENT_TOPIC,
        value={
            "resourceId": plan.resource_id,
            "version": plan.version,
            "content": markdown,
            "plainText": markdown,
            "updatedBy": [plan.user_id],
        },
    )
