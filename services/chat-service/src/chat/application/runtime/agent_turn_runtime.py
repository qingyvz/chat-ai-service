from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from beanie import PydanticObjectId
from fastapi import BackgroundTasks

from jsonschema import Draft202012Validator, SchemaError, ValidationError

from common.logger import error
from common.core.exceptions import ServiceException
from common.kafka.producer import KafkaProducerClient
from chat.domain.interfaces.llm import TextCompletionProvider
from chat.domain.interfaces.memory import MemoryProvider
from chat.domain.error_codes import ChatErrorCode
from chat.domain.repositories import SessionRepository, MessageRepository, HotContextRepository, ModelRepository, ProviderRepository
from chat.core.persistence import RedisPlanCache
from chat.service_client import AIAssetClient
from chat.application.agents import AgentResolver
from chat.application.events import ErrorEvent, StreamEvent
from chat.application.chat_context_assembler import ChatContextAssembler
from chat.application.chat_turn_finalizer import SessionTurnFinalizer
from chat.application.llm_provider_resolver import LLMProviderResolver
from chat.application.tools.skill_tools.utils.skill_matcher import SkillMatcher
from chat.application.tools.core import ToolRegistry
from chat.application.orchestration import OrchestrationContext, RawMaterials, StrategyFactory
from chat.application.token_counter import TokenCounter
from chat.application.runtime.agent_provider import SessionAgentProvider
from chat.application.runtime.context_provider import SessionContextProvider
from chat.application.runtime.model_resolver import SessionModelResolver
from chat.application.runtime.tool_scope_provider import ToolScopeProvider


def _merge_runtime_options(defaults: dict, overrides: dict) -> dict:
    result = dict(defaults or {})
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_runtime_options(result[key], value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class RuntimeServices:
    """与环境无关的共享服务一坨；根轮装配一次，子任务轮经 tool_context 透传给 call_subagent 复用"""
    tool_scope_provider: ToolScopeProvider
    assembler: ChatContextAssembler
    strategy_factory: StrategyFactory
    llm_resolver: LLMProviderResolver
    finalizer: SessionTurnFinalizer


class AgentTurnRuntime:
    """
    Agent 轮编排模板：按 agent → model → context → 能力 顺序备料，交策略跑循环，再交 finalizer 扫尾。
    构造 = 共享服务一坨（services）+ 随环境替换的 agent/model/context 三件；call_subagent 拿 services 自行换三件组装子轮。
    handle_chat 产出领域 StreamEvent；SSE 翻译留给 API 层（便于作为子 runtime 嵌套，事件只翻译一次）。
    """

    def __init__(
        self,
        *,
        services: RuntimeServices,
        agent_provider,
        model_resolver,
        context_provider,
    ) -> None:
        self._services = services
        self._agent_provider = agent_provider
        self._model_resolver = model_resolver
        self._context_provider = context_provider

    async def handle_chat(
        self,
        *,
        user_id: str,
        session_id: str,
        user_query: str,
        model_id: Optional[PydanticObjectId] = None,
        provider_id: Optional[PydanticObjectId] = None,
        frontend_states: Optional[List[Dict[str, Any]]] = None,
        user_defined_allow_tool_names: Optional[Set[str]] = None,
        user_defined_deny_tool_names: Optional[Set[str]] = None,
        user_defined_on_demand_skill_ids: Optional[Set[str]] = None,
        user_defined_force_enabled_skill_ids: Optional[Set[str]] = None,
        think_type_override: Optional[str] = None,
        plan_action: Optional[str] = None,
        plan_feedback: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None,
    ) -> AsyncIterator[StreamEvent]:
        # agent → 取 spec
        agent_info = await self._agent_provider.resolve(session_id, user_id)
        spec = agent_info.spec

        # model → 解析模型与 token 预算（子任务轮由 InheritedModelResolver 继承父模型）
        model, prompt_budget_tokens = await self._model_resolver.resolve(
            user_id=user_id, model_id=model_id, provider_id=provider_id, model_policy=spec.model_policy,
        )

        # runtime_options 校验：按 provider manifest 合并默认值并 jsonschema 校验
        manifest = self._services.llm_resolver.runtime_options_manifest(model.provider_type)
        runtime_options = _merge_runtime_options(manifest.get("defaults") or {}, model.runtime_options or {})
        try:
            Draft202012Validator.check_schema(manifest["json_schema"])
            Draft202012Validator(manifest["json_schema"]).validate(runtime_options)
        except (SchemaError, ValidationError) as exc:
            raise ServiceException(ChatErrorCode.MODEL_RUNTIME_OPTIONS_INVALID, custom_msg=str(exc))
        model = model.with_runtime_options(runtime_options)

        # context → 取记忆原料（子任务轮为隔离上下文）
        session_context = await self._context_provider.load(
            user_id=user_id, session_id=session_id, user_query=user_query,
            memory_policy=spec.memory_policy, prompt_budget_tokens=prompt_budget_tokens,
        )

        # 能力 → skill 匹配 + 派生 tool_scope；把共享服务/父模型/父 spec 注入 tool_context 供 call_subagent 自行组装子轮
        runtime_context: Dict[str, Any] = {
            "runtime_services": self._services,
            "parent_model": model,
            "parent_agent_spec": spec,
        }
        tool_scope, available_skills = await self._services.tool_scope_provider.resolve(
            session_id=session_id, user_id=user_id, user_query=user_query,
            tool_and_skill_policy=spec.tool_and_skill_policy,
            session_summary=session_context.session_summary,
            user_defined_allow_tool_names=user_defined_allow_tool_names,
            user_defined_deny_tool_names=user_defined_deny_tool_names,
            user_defined_on_demand_skill_ids=user_defined_on_demand_skill_ids,
            runtime_context=runtime_context,
        )

        # 组装编排上下文（入口组装时机下放给策略）
        ctx = OrchestrationContext(
            session_id=session_id,
            user_id=user_id,
            agent_info=agent_info,
            model=model,
            raw_materials=RawMaterials(
                user_query=user_query,
                system_prompt=spec.system_prompt,
                history_messages=session_context.history_messages,
                relevant_facts=session_context.relevant_facts,
                session_summary=session_context.session_summary,
                frontend_states=frontend_states,
                available_skills=available_skills,
            ),
            assembler=self._services.assembler,
            tool_scope=tool_scope,
            plan_action=plan_action,
            plan_feedback=plan_feedback,
        )

        # 由 think_type 选策略并跑循环；override 优先于 agent spec；runtime 只透传事件 + 事后交 finalizer
        strategy = self._services.strategy_factory.create(think_type_override or spec.think_policy.think_type)
        try:
            async for event in strategy.run(ctx):
                yield event
        except ServiceException as exc:
            error("chat stream generation failed.", session_id=session_id, exc=exc)
            yield ErrorEvent(error_text=str(exc))
            return

        # 扫尾（Session 异步登记 background_tasks；SubAgent inline 写 Redis transcript）
        await self._services.finalizer.finalize(
            background_tasks=background_tasks,
            user_id=user_id,
            session_id=session_id,
            user_query=user_query,
            agent_spec=spec,
            resolved_model=model,
            usage_tokens=ctx.usage_tokens,
            chat_record_messages=ctx.record_messages,
            windowed_history_messages=session_context.windowed_history_messages,
            session_summary=session_context.session_summary,
        )


def build_session_runtime(
    *,
    llm_resolver: LLMProviderResolver,
    text_provider: TextCompletionProvider,
    memory: MemoryProvider,
    model_repo: ModelRepository,
    provider_repo: ProviderRepository,
    session_repo: SessionRepository,
    message_repo: MessageRepository,
    hot_context_repo: HotContextRepository,
    tool_registry: ToolRegistry,
    kafka_producer: KafkaProducerClient,
    skill_matcher: SkillMatcher,
    ai_asset_client: AIAssetClient,
    plan_cache: RedisPlanCache,
    token_counter: TokenCounter,
    agent_resolver: AgentResolver | None = None,
) -> AgentTurnRuntime:
    """装配会话根轮：共享服务一坨（子任务轮由 call_subagent 复用）+ Session* 三件"""
    services = RuntimeServices(
        tool_scope_provider=ToolScopeProvider(skill_matcher, tool_registry),
        assembler=ChatContextAssembler(),
        strategy_factory=StrategyFactory(llm_resolver, token_counter, text_provider, ai_asset_client, plan_cache),
        llm_resolver=llm_resolver,
        finalizer=SessionTurnFinalizer(
            llm=text_provider, memory=memory,
            message_repo=message_repo, session_repo=session_repo, hot_context_repo=hot_context_repo,
            provider_repo=provider_repo, kafka_producer=kafka_producer,
        ),
    )
    return AgentTurnRuntime(
        services=services,
        agent_provider=SessionAgentProvider(agent_resolver, session_repo),
        model_resolver=SessionModelResolver(model_repo),
        context_provider=SessionContextProvider(memory, message_repo, session_repo, hot_context_repo),
    )
