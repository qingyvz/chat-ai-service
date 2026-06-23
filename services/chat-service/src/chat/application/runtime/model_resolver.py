from typing import Optional, Protocol, Tuple

from beanie import PydanticObjectId

from chat.application.agents import AgentModelPolicy
from chat.core.config.app_settings import settings
from chat.domain.repositories import ModelRepository
from chat.domain.repositories.model_repo import ModelRequestInfo


def prompt_budget_for(model: ModelRequestInfo) -> int:
    """由模型上下文窗口与输出预留算出 prompt token 预算"""
    context_limit = model.context_window_tokens or settings.CTX_TOKEN_LIMIT
    output_reserve = model.max_output_tokens or settings.CTX_DEFAULT_OUTPUT_RESERVE_TOKENS
    return max(context_limit - output_reserve, settings.CTX_MIN_PROMPT_BUDGET_TOKENS)


class ModelResolver(Protocol):
    """取模型 + token 预算；环境可换：Session 按请求/策略解析，SubAgent 继承父模型"""
    async def resolve(
        self,
        user_id: str,
        model_id: Optional[PydanticObjectId],
        provider_id: Optional[PydanticObjectId],
        model_policy: AgentModelPolicy,
    ) -> Tuple[ModelRequestInfo, int]:
        ...


class SessionModelResolver:
    """会话场景：解析模型/映射/供应商/凭证，并算出 prompt token 预算"""

    def __init__(self, model_repo: ModelRepository) -> None:
        self._model_repo = model_repo

    async def resolve(
        self,
        user_id: str,
        model_id: Optional[PydanticObjectId],
        provider_id: Optional[PydanticObjectId],
        model_policy: AgentModelPolicy,
    ) -> Tuple[ModelRequestInfo, int]:
        resolved_model_id = model_id
        resolved_provider_id = provider_id

        # 如果禁止覆盖，且 agent 指定了默认模型和供应商
        if not model_policy.allow_request_override:
            if model_policy.default_model_id: resolved_model_id = PydanticObjectId(model_policy.default_model_id)
            if model_policy.default_provider_id: resolved_provider_id = PydanticObjectId(model_policy.default_provider_id)

        resolved_model = await self._model_repo.resolve_model_for_chat(
            model_id=resolved_model_id,
            user_id=user_id,
            provider_id=resolved_provider_id,
        )
        return resolved_model, prompt_budget_for(resolved_model)


class InheritedModelResolver:
    """子任务场景：继承父轮已解析的模型，跳过模型解析"""

    def __init__(self, parent_model: ModelRequestInfo) -> None:
        self._parent_model = parent_model

    async def resolve(
        self,
        user_id: str,
        model_id: Optional[PydanticObjectId],
        provider_id: Optional[PydanticObjectId],
        model_policy: AgentModelPolicy,
    ) -> Tuple[ModelRequestInfo, int]:
        return self._parent_model, prompt_budget_for(self._parent_model)
