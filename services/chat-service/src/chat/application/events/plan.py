from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chat.application.events.base import StreamEvent


@dataclass(frozen=True)
class PlanCreatedEvent(StreamEvent):
    """整张 to-do list 生成完毕（走 AI SDK data-plan 自定义数据块）"""
    plan_id: str
    steps: List[Dict[str, Any]] = field(default_factory=list)  # [{"id","title","status"}]


@dataclass(frozen=True)
class PlanStepStatusEvent(StreamEvent):
    """单个步骤状态流转（data-plan-step）"""
    step_id: str
    status: str
    result_summary: Optional[str] = None


@dataclass(frozen=True)
class PlanUpdatedEvent(StreamEvent):
    """重规划后覆盖渲染整张计划（复用 data-plan）"""
    plan_id: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
