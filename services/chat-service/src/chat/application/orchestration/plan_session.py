import asyncio
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from chat.domain.entities import Plan, PlanStep
from chat.application.events import PlanStepStatusEvent, PlanUpdatedEvent, StreamEvent

if TYPE_CHECKING:
    from chat.application.orchestration.step_executor import StepExecutor

_ACTIONABLE = ("pending", "failed")


class PlanSession:
    """Plan-and-Execute DAG 的 turn 级共享状态 + 重规划自动机：策略与 plan 工具共享一个实例"""

    def __init__(self, plan: Plan, max_attempts: int, max_parallel: int) -> None:
        self.plan = plan
        self.max_attempts = max_attempts
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.executor: Optional["StepExecutor"] = None  # 策略 build 后回填
        self.step_tokens = 0
        self._steps: Dict[str, PlanStep] = {step.step_id: step for step in plan.steps}
        self._attempts: Dict[str, int] = {}
        self._terminal: set[str] = set()           # 终态失败（弃步）的 step_id
        self._pending_events: List[StreamEvent] = []
        self._replan_queue: List[Tuple[str, str]] = []

    # --- 查询 ---
    def get_step(self, step_id: str) -> Optional[PlanStep]:
        return self._steps.get(step_id)

    def is_ready(self, step: PlanStep) -> bool:
        """未弃步、自身可执行、且所有依赖已完成"""
        if step.step_id in self._terminal or step.status not in _ACTIONABLE:
            return False
        return all(self._steps.get(d) is not None and self._steps[d].status == "completed" for d in step.depends_on)

    def dep_results(self, step: PlanStep) -> List[Tuple[str, str]]:
        return [(self._steps[d].title, self._steps[d].result_summary or "") for d in step.depends_on if d in self._steps]

    def all_done(self) -> bool:
        """所有步要么 completed 要么终态失败 → 整轮可收尾"""
        return all(step.status == "completed" or step.step_id in self._terminal for step in self.plan.steps)

    def steps_payload(self) -> List[Dict[str, str]]:
        return [{"id": step.step_id, "title": step.title, "status": step.status} for step in self.plan.steps]

    # --- 状态流转（事件推进 _pending_events，由策略 drain 后 yield）---
    def start(self, step_id: str) -> None:
        step = self._steps.get(step_id)
        if step is not None:
            step.status = "in_progress"

    def mark_completed(self, step_id: str, summary: str) -> None:
        step = self._steps.get(step_id)
        if step is None:
            return
        step.status = "completed"
        step.result_summary = summary
        self._pending_events.append(PlanStepStatusEvent(step_id=step_id, status="completed", result_summary=summary))

    def mark_failed(self, step_id: str, error: str) -> int:
        """记一次失败，返回累计 attempts；超预算自动弃步（终态 + 剪下游），否则排队重规划"""
        step = self._steps.get(step_id)
        if step is None:
            return 0
        attempts = self._attempts.get(step_id, 0) + 1
        self._attempts[step_id] = attempts
        if attempts >= self.max_attempts:
            self.abandon(step_id, f"已达重规划上限({self.max_attempts})，放弃：{error}")
        else:
            step.status = "failed"
            self._pending_events.append(PlanStepStatusEvent(step_id=step_id, status="failed", result_summary=error))
            self._replan_queue.append((step_id, error))
        return attempts

    def abandon(self, step_id: str, reason: str) -> None:
        """显式/兜底弃步：标终态失败并递归剪掉下游依赖步"""
        step = self._steps.get(step_id)
        if step is None or step_id in self._terminal:
            return
        self._terminal.add(step_id)
        step.status = "failed"
        step.result_summary = reason
        self._pending_events.append(PlanStepStatusEvent(step_id=step_id, status="failed", result_summary=reason))
        self._prune_dependents(step_id)

    def _prune_dependents(self, step_id: str) -> None:
        for step in self.plan.steps:
            if step_id in step.depends_on and step.step_id not in self._terminal and step.status != "completed":
                self._terminal.add(step.step_id)
                step.status = "failed"
                step.result_summary = f"依赖步骤 {step_id} 失败，跳过"
                self._pending_events.append(PlanStepStatusEvent(step_id=step.step_id, status="failed", result_summary=step.result_summary))
                self._prune_dependents(step.step_id)

    def replace_steps(self, raw_steps: List[Dict[str, Any]]) -> None:
        """update_plan 用：按传入步重建计划，保留已 completed 步的结果，新增/改动步置 pending"""
        new_steps: List[PlanStep] = []
        new_index: Dict[str, PlanStep] = {}
        for raw in raw_steps:
            title = (raw.get("title") or "").strip()
            if not title:
                continue
            step_id = (raw.get("id") or "").strip()
            existing = self._steps.get(step_id) if step_id else None
            if existing is not None and existing.status == "completed":
                step = existing  # 保留已完成结果
            else:
                step = PlanStep(
                    step_id=step_id or f"step_{uuid.uuid4().hex[:6]}",
                    title=title,
                    description=(raw.get("description") or "").strip(),
                    depends_on=list(raw.get("depends_on") or []),
                )
                self._attempts.pop(step.step_id, None)
                self._terminal.discard(step.step_id)
            new_steps.append(step)
            new_index[step.step_id] = step
        self.plan.steps = new_steps
        self._steps = new_index
        self._pending_events.append(PlanUpdatedEvent(plan_id=self.plan.plan_id, steps=self.steps_payload()))

    # --- 自动机 / drain ---
    def drain_events(self) -> List[StreamEvent]:
        events = self._pending_events
        self._pending_events = []
        return events

    def take_replans(self) -> List[Tuple[str, str]]:
        queue = self._replan_queue
        self._replan_queue = []
        return queue
