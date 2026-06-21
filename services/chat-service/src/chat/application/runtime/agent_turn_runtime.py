import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

from beanie import PydanticObjectId
from fastapi import BackgroundTasks

from common.logger import error
from common.core.exceptions import ServiceException
from common.kafka.producer import KafkaProducerClient
from chat.domain.interfaces.llm import LLMProvider
from chat.domain.interfaces.memory import MemoryProvider
from chat.domain.repositories import SessionRepository, MessageRepository, HotContextRepository, ModelRepository, ProviderRepository
from chat.domain.repositories.model_repo import ModelRequestInfo
from chat.application.agents import AgentResolver, AgentSpec, SubAgentRepository
from chat.application.events import ErrorEvent, StepFinishEvent, StreamEvent
from chat.application.chat_context_assembler import ChatContextAssembler
from chat.application.chat_turn_finalizer import SessionTurnFinalizer
from chat.application.tools.skill_tools.utils.skill_matcher import SkillMatcher
from chat.application.tools.core import ToolRegistry
from chat.application.orchestration import AgentStepRunner, OrchestrationContext, RawMaterials, StrategyFactory
from chat.application.runtime.agent_provider import SessionAgentProvider, SubAgentProvider, build_subagent_info
from chat.application.runtime.context_provider import SessionContextProvider, SubAgentContextProvider
from chat.application.runtime.model_resolver import SessionModelResolver, InheritedModelResolver
from chat.application.runtime.tool_scope_provider import ToolScopeProvider


class AgentTurnRuntime:
    """
    Agent 轮编排模板：按 agent → model → context → 能力 顺序备料，交策略跑循环，再交 finalizer 扫尾。
    根轮注入 Session* 三件 + subagent_spawner；子任务轮注入 SubAgent* 三件、spawner=None，模板不变。
    handle_chat 产出领域 StreamEvent；SSE 翻译留给 API 层（便于作为子 runtime 嵌套，事件只翻译一次）。
    """

    def __init__(
        self,
        *,
        agent_provider,
        model_resolver,
        context_provider,
        tool_scope_provider: ToolScopeProvider,
        assembler: ChatContextAssembler,
        strategy_factory: StrategyFactory,
        finalizer,
        subagent_spawner: Optional["SubAgentSpawner"] = None,
    ) -> None:
        self._agent_provider = agent_provider
        self._model_resolver = model_resolver
        self._context_provider = context_provider
        self._tool_scope_provider = tool_scope_provider
        self._assembler = assembler
        self._strategy_factory = strategy_factory
        self._finalizer = finalizer
        self._subagent_spawner = subagent_spawner

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
        background_tasks: Optional[BackgroundTasks] = None,
    ) -> AsyncIterator[StreamEvent]:
        # agent → 取 spec
        agent_info = await self._agent_provider.resolve(session_id, user_id)
        spec = agent_info.spec

        # model → 解析模型与 token 预算（子任务轮由 InheritedModelResolver 继承父模型）
        model, prompt_budget_tokens = await self._model_resolver.resolve(
            user_id=user_id, model_id=model_id, provider_id=provider_id, model_policy=spec.model_policy,
        )

        # context → 取记忆原料（子任务轮为隔离上下文）
        session_context = await self._context_provider.load(
            user_id=user_id, session_id=session_id, user_query=user_query,
            memory_policy=spec.memory_policy, prompt_budget_tokens=prompt_budget_tokens,
        )

        # 能力 → skill 匹配 + 派生 tool_scope；把 spawner/父模型/父 spec 注入 tool_context 供 subagent 工具读取
        runtime_context: Dict[str, Any] = {
            "subagent_spawner": self._subagent_spawner,
            "parent_model": model,
            "parent_agent_spec": spec,
        }
        tool_scope, available_skills = await self._tool_scope_provider.resolve(
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
            assembler=self._assembler,
            tool_scope=tool_scope,
        )

        # 由 think_type 选策略并跑循环；override 优先于 agent spec；runtime 只透传事件 + 事后交 finalizer
        strategy = self._strategy_factory.create(think_type_override or spec.think_policy.think_type)
        try:
            async for event in strategy.run(ctx):
                yield event
        except ServiceException as exc:
            error("chat stream generation failed.", session_id=session_id, exc=exc)
            yield ErrorEvent(error_text=str(exc))
            return

        # 扫尾（Session 异步登记 background_tasks；SubAgent inline 写 Redis transcript）
        await self._finalizer.finalize(
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


class SubAgentSpawner:
    """承载 subagent 机制：create 落库 spec，call 跑子 runtime 到结束并返回结论；被 create_subagent/call_subagent 两个 tool 调用"""

    def __init__(
        self,
        *,
        step_runner: AgentStepRunner,
        llm: LLMProvider,
        assembler: ChatContextAssembler,
        tool_scope_provider: ToolScopeProvider,
        subagent_repo: SubAgentRepository,
        finalizer: SessionTurnFinalizer,
    ) -> None:
        self._assembler = assembler
        self._tool_scope_provider = tool_scope_provider
        self._subagent_repo = subagent_repo
        self._finalizer = finalizer  # subagent 复用同一个 finalizer（background_tasks=None → inline 计费）
        self._react_factory = StrategyFactory(step_runner, llm)

    async def create(self, *, session_id: str, parent_spec: AgentSpec, role: str) -> str:
        """造一个收窄的 subagent spec 落 Redis，返回 subagent_id"""
        subagent_id = f"subagent_{uuid.uuid4().hex[:8]}"
        await self._subagent_repo.save(
            session_id, subagent_id, build_subagent_info(parent_spec, role, session_id, subagent_id),
        )
        return subagent_id

    async def call(
        self,
        *,
        session_id: str,
        user_id: str,
        subagent_id: str,
        parent_model: ModelRequestInfo,
        task: str,
        prior_results: List[Tuple[str, str]],
    ) -> str:
        """用同一 AgentTurnRuntime 模板（SubAgent* 三件）跑子任务到结束，返回最终结论文本"""
        sub_runtime = AgentTurnRuntime(
            agent_provider=SubAgentProvider(self._subagent_repo, subagent_id),
            model_resolver=InheritedModelResolver(parent_model),
            context_provider=SubAgentContextProvider(prior_results),
            tool_scope_provider=self._tool_scope_provider,
            assembler=self._assembler,
            strategy_factory=self._react_factory,
            finalizer=self._finalizer,
            subagent_spawner=None,
        )
        final_text = ""
        async for event in sub_runtime.handle_chat(
            user_id=user_id, session_id=session_id, user_query=task, background_tasks=None,
        ):
            if isinstance(event, StepFinishEvent) and event.is_finished and event.final_assistant_message is not None:
                final_text = event.final_assistant_message.content or final_text
        return final_text


def build_session_runtime(
    *,
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
    subagent_repo: SubAgentRepository,
    agent_resolver: AgentResolver | None = None,
) -> AgentTurnRuntime:
    """装配会话根轮：Session* 三件 + 共享服务 + subagent_spawner（供 subagent 工具调用）"""
    assembler = ChatContextAssembler()
    tool_scope_provider = ToolScopeProvider(skill_matcher, tool_registry)
    step_runner = AgentStepRunner(llm)
    finalizer = SessionTurnFinalizer(
        llm=llm, memory=memory,
        message_repo=message_repo, session_repo=session_repo, hot_context_repo=hot_context_repo,
        provider_repo=provider_repo, kafka_producer=kafka_producer,
    )
    spawner = SubAgentSpawner(
        step_runner=step_runner, llm=llm, assembler=assembler,
        tool_scope_provider=tool_scope_provider, subagent_repo=subagent_repo,
        finalizer=finalizer,  # subagent 复用同一个 finalizer
    )
    return AgentTurnRuntime(
        agent_provider=SessionAgentProvider(agent_resolver, session_repo),
        model_resolver=SessionModelResolver(model_repo),
        context_provider=SessionContextProvider(memory, message_repo, session_repo, hot_context_repo),
        tool_scope_provider=tool_scope_provider,
        assembler=assembler,
        strategy_factory=StrategyFactory(step_runner, llm),
        finalizer=finalizer,
        subagent_spawner=spawner,
    )
