from typing import Optional

from chat.domain.interfaces import LLMProvider
from chat.application.orchestration.base import OrchestrationStrategy
from chat.application.orchestration.react import ReActStrategy
from chat.application.orchestration.plan_execute import PlanAndExecuteStrategy
from chat.application.orchestration.step_runner import AgentStepRunner

_PLAN_AND_EXECUTE = "PlanAndExecute"


class StrategyFactory:
    """按 think_type 选编排策略；未知/空默认落 ReAct（保持现状）"""

    def __init__(self, step_runner: AgentStepRunner, llm: LLMProvider, sub_runtime) -> None:
        self._step_runner = step_runner
        self._llm = llm
        self._sub_runtime = sub_runtime  # SubAgentTurnRuntime（Plan-Execute Executor 派子 runtime 用）

    def create(self, think_type: Optional[str]) -> OrchestrationStrategy:
        if think_type == _PLAN_AND_EXECUTE:
            return PlanAndExecuteStrategy(self._llm, self._sub_runtime)
        return ReActStrategy(self._step_runner)
