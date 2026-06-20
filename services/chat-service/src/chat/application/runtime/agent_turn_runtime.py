from typing import Optional, List, Dict, Any, Set

from beanie import PydanticObjectId
from fastapi import BackgroundTasks

from common.logger import error

from chat.domain.interfaces.llm import LLMProvider
from chat.domain.interfaces.memory import MemoryProvider
from chat.domain.repositories import SessionRepository, MessageRepository, HotContextRepository, ModelRepository, ProviderRepository
from common.core.exceptions import ServiceException
from chat.application.chat_context_assembler import ChatContextAssembler
from chat.application.chat_turn_finalizer import SessionTurnFinalizer
from chat.application.agents import AgentResolver
from chat.application.events import ErrorEvent
from chat.api.vercel_sse_mapper import to_vercel_sse
from chat.application.tools.skill_tools.utils.skill_matcher import SkillMatcher
from chat.application.tools.core import ToolRegistry
from common.kafka.producer import KafkaProducerClient
from chat.application.orchestration import AgentStepRunner, OrchestrationContext, RawMaterials, StrategyFactory
from chat.application.runtime.agent_provider import SessionAgentProvider
from chat.application.runtime.context_provider import SessionContextProvider
from chat.application.runtime.model_resolver import ModelResolver
from chat.application.runtime.tool_scope_provider import ToolScopeProvider


class AgentTurnRuntime:
    """
    Agent 轮编排模板：按 agent → model → context → 能力 顺序备料，交策略跑循环，再交 finalizer 扫尾。
    环境可换三件（AgentProvider / ContextProvider / TurnFinalizer）+ 共享服务（ModelResolver / ToolScopeProvider / 组装工具箱）。
    """

    def __init__(
            self,
            llm: LLMProvider,
            memory: MemoryProvider,
            model_repo: ModelRepository,
            provider_repo: ProviderRepository,
            session_repo: SessionRepository,
            message_repo: MessageRepository,
            hot_context_repo: HotContextRepository,
            tool_registry: ToolRegistry,
            kafka_producer: KafkaProducerClient,
            skill_matcher: SkillMatcher,
            agent_resolver: AgentResolver | None = None,
    ):
        self._assembler = ChatContextAssembler(
            message_repo=message_repo, session_repo=session_repo, hot_context_repo=hot_context_repo
        )
        # 环境可换三件（当前仅 Session 实现）
        self._agent_provider = SessionAgentProvider(agent_resolver, session_repo)
        self._context_provider = SessionContextProvider(memory, self._assembler)
        self._finalizer = SessionTurnFinalizer(
            llm=llm, memory=memory,
            message_repo=message_repo, session_repo=session_repo, hot_context_repo=hot_context_repo,
            provider_repo=provider_repo,
            kafka_producer=kafka_producer,
        )
        # 共享服务
        self._model_resolver = ModelResolver(model_repo)
        self._tool_scope_provider = ToolScopeProvider(skill_matcher, tool_registry)
        self._strategy_factory = StrategyFactory(AgentStepRunner(llm), llm)

    # -------------------------------------------------------------------------
    # 公共入口
    # -------------------------------------------------------------------------
    async def handle_chat(
            self,
            user_id: str,
            session_id: str,
            user_query: str,
            background_tasks: BackgroundTasks,
            model_id: PydanticObjectId,
            provider_id: Optional[PydanticObjectId] = None,
            frontend_states: Optional[List[Dict[str, Any]]] = None,
            user_defined_allow_tool_names: Optional[Set[str]] = None,
            user_defined_deny_tool_names: Optional[Set[str]] = None,
            user_defined_on_demand_skill_ids: Optional[Set[str]] = None,
            user_defined_force_enabled_skill_ids: Optional[Set[str]] = None,
            think_type_override: Optional[str] = None,
    ):
        # agent → 取 spec
        agent = await self._agent_provider.resolve(session_id, user_id)
        agent_spec = agent.spec

        # model → 解析模型与 token 预算
        resolved_model, prompt_budget_tokens = await self._model_resolver.resolve(
            user_id=user_id,
            model_id=model_id,
            provider_id=provider_id,
            model_policy=agent_spec.model_policy,
        )

        # context → 取记忆原料
        session_context = await self._context_provider.load(
            user_id=user_id,
            session_id=session_id,
            user_query=user_query,
            memory_policy=agent_spec.memory_policy,
            prompt_budget_tokens=prompt_budget_tokens,
        )

        # 能力 → skill 匹配 + 派生 tool_scope（tool_scope 依赖 context 是否有 summary）
        tool_scope, available_skills = await self._tool_scope_provider.resolve(
            session_id=session_id,
            user_id=user_id,
            user_query=user_query,
            tool_and_skill_policy=agent_spec.tool_and_skill_policy,
            session_summary=session_context.session_summary,
            user_defined_allow_tool_names=user_defined_allow_tool_names,
            user_defined_deny_tool_names=user_defined_deny_tool_names,
            user_defined_on_demand_skill_ids=user_defined_on_demand_skill_ids,
        )

        # 组装编排上下文（入口组装时机下放给策略）
        ctx = OrchestrationContext(
            session_id=session_id,
            user_id=user_id,
            agent_spec=agent_spec,
            model=resolved_model,
            raw_materials=RawMaterials(
                user_query=user_query,
                system_prompt=agent_spec.system_prompt,
                history_messages=session_context.history_messages,
                relevant_facts=session_context.relevant_facts,
                session_summary=session_context.session_summary,
                frontend_states=frontend_states,
                available_skills=available_skills,
            ),
            assembler=self._assembler,
            tool_scope=tool_scope,
        )

        # 由 think_type 选策略并跑循环；override 优先于 agent spec；runtime 只透传事件 + 事后把 ctx 交给 finalizer
        strategy = self._strategy_factory.create(think_type_override or agent_spec.think_policy.think_type)
        try:
            async for event in strategy.run(ctx):
                yield to_vercel_sse(event)
        except ServiceException as e:
            error("chat stream generation failed.", session_id=session_id, exc=e)
            yield to_vercel_sse(ErrorEvent(error_text=str(e)))
            return

        # 扫尾
        await self._finalizer.finalize(
            background_tasks=background_tasks,
            user_id=user_id,
            session_id=session_id,
            user_query=user_query,
            agent_spec=agent_spec,
            resolved_model=resolved_model,
            usage_tokens=ctx.usage_tokens,
            chat_record_messages=ctx.record_messages,
            windowed_history_messages=session_context.windowed_history_messages,
            session_summary=session_context.session_summary,
        )
