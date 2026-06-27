from typing import Optional

from chat.domain.interfaces.llm import TextCompletionProvider
from chat.domain.repositories import PlanRepository
from chat.application.llm_provider_resolver import LLMProviderResolver
from chat.application.token_counter import TokenCounter
from chat.application.orchestration.base import OrchestrationStrategy
from chat.application.orchestration.react import ReActStrategy
from chat.application.orchestration.plan_mode import PlanModeStrategy
from chat.application.orchestration.step_runner import ReActStepRunner

_REACT = "ReAct"
_PLAN_MODE = "PlanMode"


class StrategyFactory:
    """按 think_type 选编排策略，并为各策略现造其专属 step_runner；未知/空默认落 ReAct"""

    def __init__(self, llm_resolver: LLMProviderResolver, token_counter: TokenCounter, text_provider: TextCompletionProvider, plan_repo: PlanRepository) -> None:
        self._llm_resolver = llm_resolver
        self._token_counter = token_counter
        self._text_provider = text_provider
        self._plan_repo = plan_repo

    def create(self, think_type: Optional[str]) -> OrchestrationStrategy:
        if think_type == _PLAN_MODE:
            return PlanModeStrategy(
                step_runner=ReActStepRunner(self._llm_resolver, self._token_counter),
                plan_repo=self._plan_repo,
            )
        # 默认/未知 → ReAct
        return ReActStrategy(step_runner=ReActStepRunner(self._llm_resolver, self._token_counter))
