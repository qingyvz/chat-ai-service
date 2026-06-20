from typing import Optional

from chat.application.orchestration.base import OrchestrationStrategy
from chat.application.orchestration.react import ReActStrategy
from chat.application.orchestration.step_runner import AgentStepRunner


class StrategyFactory:
    """按 think_type 选编排策略；当前仅 ReAct，未知/空默认落 ReAct（保持现状）"""

    def __init__(self, step_runner: AgentStepRunner) -> None:
        self._step_runner = step_runner

    def create(self, think_type: Optional[str]) -> OrchestrationStrategy:
        return ReActStrategy(self._step_runner)
