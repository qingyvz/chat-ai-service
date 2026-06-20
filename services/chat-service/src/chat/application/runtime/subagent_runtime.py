from typing import AsyncIterator, Dict, List, Tuple

from common.logger import warn
from chat.core.config.app_settings import settings
from chat.domain.entities import ChatMessage, PlanStep
from chat.application.agents import (
    Agent,
    AgentMemoryPolicy,
    AgentSpec,
    AgentToolAndSkillPolicy,
    SubAgentRepository,
)
from chat.application.chat_context_assembler import ChatContextAssembler
from chat.application.events import StepFinishEvent, StreamEvent
from chat.application.orchestration.base import OrchestrationContext
from chat.application.orchestration.step_runner import AgentStepRunner
from chat.application.runtime.tool_scope_provider import ToolScopeProvider

_SUBAGENT_SYSTEM_PROMPT = (
    "You are an execution sub-agent dispatched by an orchestrator. Complete ONLY the single subtask in "
    "<subtask>, using any prior results in <completed_steps> as context. Be concise and produce a direct, "
    "usable result. Do not plan further; do not delegate."
)


class SubAgentTurnRuntime:
    """
    子 runtime（同 ReAct 内核，SubAgent* 语义）：造 subagent spec 存 Redis → 隔离上下文 → 收窄 tool_scope →
    继承父模型跑 ReAct → inline 写 Redis transcript（不计费，usage 上卷父 ctx）。subagent 不再生 subagent。
    """

    def __init__(
        self,
        step_runner: AgentStepRunner,
        tool_scope_provider: ToolScopeProvider,
        assembler: ChatContextAssembler,
        subagent_repo: SubAgentRepository,
    ) -> None:
        self._step_runner = step_runner
        self._tool_scope_provider = tool_scope_provider
        self._assembler = assembler
        self._subagent_repo = subagent_repo

    async def run(
        self,
        parent_ctx: OrchestrationContext,
        role: str,
        step: PlanStep,
        prior_results: List[Tuple[str, str]],
        holder: Dict[str, str],
    ) -> AsyncIterator[StreamEvent]:
        session_id = parent_ctx.session_id

        # 造 subagent spec（复用 AgentSpec，收窄）并存 Redis，再读回（演示 Redis-backed AgentProvider 路径）
        await self._subagent_repo.save(session_id, step.step_id, self._build_subagent(parent_ctx.agent_spec, role, session_id, step.step_id))
        agent = await self._subagent_repo.get(session_id, step.step_id)
        spec = agent.spec

        # 隔离上下文：执行角色提示 + 前序产出 + 当前子任务（不带会话历史）
        messages = self._assembler.build_executor_context(
            session_id=session_id,
            executor_system_prompt=spec.system_prompt,
            subtask_title=step.title,
            subtask_description=step.description,
            prior_results=prior_results,
        )

        # tool_scope 由 subagent 收窄 policy 派生（隔离上下文无 summary → session 工具不解禁）
        tool_scope, _ = await self._tool_scope_provider.resolve(
            session_id=session_id,
            user_id=parent_ctx.user_id,
            user_query=step.description,
            tool_and_skill_policy=spec.tool_and_skill_policy,
            session_summary=None,
        )

        # ReAct 循环（模型继承父）；transcript 记录隔离对话以 inline 落 Redis
        transcript: List[ChatMessage] = list(messages)
        max_iterations = spec.agent_max_iterations or settings.AGENT_MAX_ITERATIONS
        for iteration in range(max_iterations):
            step_finish_event = None
            async for item in self._step_runner.run(
                messages=messages,
                session_id=session_id,
                model_name=parent_ctx.model.model_name,
                model_id=parent_ctx.model.model_id,
                api_base=parent_ctx.model.api_base_url,
                api_key=parent_ctx.model.api_key,
                iteration=iteration,
                tool_scope=tool_scope,
            ):
                if isinstance(item, StepFinishEvent):
                    step_finish_event = item
                yield item

            assert step_finish_event is not None
            parent_ctx.usage_tokens += step_finish_event.usage_tokens  # usage 上卷父 ctx
            if step_finish_event.is_finished:
                holder["text"] = step_finish_event.final_assistant_message.content or ""
                transcript.append(step_finish_event.final_assistant_message)
                break
            else:
                messages.extend(step_finish_event.intermediate_messages)
                transcript.extend(step_finish_event.intermediate_messages)
        else:
            warn("subagent loop exhausted.", session_id=session_id, step_id=step.step_id)

        # inline 扫尾：只写 Redis transcript（不计费）
        await self._subagent_repo.save_transcript(session_id, step.step_id, transcript)

    @staticmethod
    def _build_subagent(parent_spec: AgentSpec, role: str, session_id: str, step_id: str) -> Agent:
        """subagent 本质就是个 Agent：执行向 system prompt + 收窄 tool/skill policy + 继承迭代上限"""
        narrowed = AgentToolAndSkillPolicy(
            enable_use_tool=parent_spec.tool_and_skill_policy.enable_use_tool,
            allow_tool_names=parent_spec.tool_and_skill_policy.allow_tool_names,
            deny_tool_names=parent_spec.tool_and_skill_policy.deny_tool_names,
            enable_use_skill=False,
        )
        spec = AgentSpec(
            system_prompt=_SUBAGENT_SYSTEM_PROMPT,
            auto_generate_title=False,
            billing_group_id=parent_spec.billing_group_id,
            agent_max_iterations=parent_spec.agent_max_iterations,
            model_policy=parent_spec.model_policy,
            tool_and_skill_policy=narrowed,
            memory_policy=AgentMemoryPolicy(
                enable_chat_memory=False,
                enable_long_term_memory=False,
                enable_chat_memory_summary=False,
            ),
        )
        return Agent(
            agent_id=f"subagent:{session_id}:{step_id}",
            name=role,
            description="plan-execute executor subagent",
            version=0,
            spec=spec,
        )
