from typing import Optional

from chat.domain.interfaces.llm import TextCompletionProvider
from chat.application.llm_provider_resolver import LLMProviderResolver
from chat.application.orchestration.base import OrchestrationStrategy
from chat.application.orchestration.react import ReActStrategy
from chat.application.orchestration.plan_execute import PlanAndExecuteStrategy
from chat.application.orchestration.step_runner import AgentStepRunner

_REACT = "ReAct"
_PLAN_AND_EXECUTE = "PlanAndExecute"


class StrategyFactory:
    """按 think_type 选编排策略；未知/空默认落 ReAct（保持现状）"""

    def __init__(self, step_runner: AgentStepRunner, text_provider: TextCompletionProvider, resolver: LLMProviderResolver) -> None:
        self._step_runner = step_runner
        self._text_provider = text_provider
        self._resolver = resolver

    def create(self, think_type: Optional[str]) -> OrchestrationStrategy:
        if think_type == _PLAN_AND_EXECUTE:
            return PlanAndExecuteStrategy(self._text_provider, self._resolver)
        if think_type == _REACT:
            return ReActStrategy(self._step_runner)
        # 默认/未知 → ReAct
        return ReActStrategy(self._step_runner)
