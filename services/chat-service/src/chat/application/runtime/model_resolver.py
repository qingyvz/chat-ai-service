from typing import Optional, Tuple

from beanie import PydanticObjectId

from chat.application.agents import AgentModelPolicy
from chat.core.config.app_settings import settings
from chat.domain.repositories import ModelRepository
from chat.domain.repositories.model_repo import ModelRequestInfo


class ModelResolver:
    """共享服务：解析模型/映射/供应商/凭证，并算出 prompt token 预算（单实现，不随环境换）"""

    def __init__(self, model_repo: ModelRepository) -> None:
        self._model_repo = model_repo

    async def resolve(
        self,
        user_id: str,
        model_id: PydanticObjectId,
        provider_id: Optional[PydanticObjectId],
        model_policy: AgentModelPolicy,
    ) -> Tuple[ModelRequestInfo, int]:
        resolved_model_id = model_id
        resolved_provider_id = provider_id

        # 如果禁止覆盖，且 agent 指定了默认模型和供应商
        if not model_policy.allow_request_override:
            if model_policy.default_model_id: resolved_model_id = PydanticObjectId(model_policy.default_model_id)
            if model_policy.default_provider_id: resolved_provider_id = PydanticObjectId(model_policy.default_provider_id)

        # 解析模型、映射、供应商和 API 凭证
        resolved_model = await self._model_repo.resolve_model_for_chat(
            model_id=resolved_model_id,
            user_id=user_id,
            provider_id=resolved_provider_id,
        )

        # Token 窗口尺寸 → prompt 预算
        context_limit = resolved_model.context_window_tokens or settings.CTX_TOKEN_LIMIT
        output_reserve = resolved_model.max_output_tokens or settings.CTX_DEFAULT_OUTPUT_RESERVE_TOKENS
        prompt_budget_tokens = max(
            context_limit - output_reserve,
            settings.CTX_MIN_PROMPT_BUDGET_TOKENS,
        )

        return resolved_model, prompt_budget_tokens
