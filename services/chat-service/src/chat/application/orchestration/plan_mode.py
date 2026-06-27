from typing import AsyncIterator, List

from chat.core.config.app_settings import settings
from chat.domain.entities import ChatMessage, Role, Plan
from chat.domain.repositories import PlanRepository
from chat.application.events import StepFinishEvent, StreamEvent
from chat.application.tools import ToolScope
from chat.application.orchestration.base import OrchestrationContext, OrchestrationStrategy
from chat.application.orchestration.plan_context import PlanContext
from chat.application.orchestration.plan_prompts import (
    EXECUTE_DIRECTIVE,
    PLAN_FORMAT_DIRECTIVE,
    change_directive,
    change_plan_block,
    execute_plan_block,
)
from chat.application.orchestration.step_runner import ReActStepRunner


class PlanModeStrategy(OrchestrationStrategy):
    """PlanMode（人审 plan 文件 + ReAct 执行）：按持久化 plan 状态 + 请求意图分支——无 plan 草拟 / execute 执行 / change 重规划"""

    def __init__(self, step_runner: ReActStepRunner, plan_repo: PlanRepository) -> None:
        self._step_runner = step_runner
        self._plan_repo = plan_repo

    async def run(self, ctx: OrchestrationContext) -> AsyncIterator[StreamEvent]:
        rm = ctx.raw_materials
        ctx.record_messages.append(ChatMessage(
            session_id=ctx.session_id,
            role=Role.USER,
            content=rm.user_query,
            metadata={
                "relevant_facts": rm.relevant_facts,
                "frontend_states": rm.frontend_states or {},
                "available_skills_id": [skill.skill_id for skill in rm.available_skills] or [],
            },
        ))

        max_iterations = ctx.agent_info.spec.agent_max_iterations or settings.AGENT_MAX_ITERATIONS
        plan_ctx = PlanContext()
        scope = ctx.tool_scope.bind("plan_context", plan_ctx)
        active = await self._plan_repo.get_active_for_session(ctx.session_id, ctx.user_id)

        # 1. 无 plan → 草拟：强制模型调 create_file 产出 plan，待审查
        if active is None:
            messages = self._assemble(ctx)
            messages.append(ChatMessage(session_id=ctx.session_id, role=Role.USER, content=PLAN_FORMAT_DIRECTIVE))
            async for event in self._drive(ctx, messages, scope, plan_ctx, max_iterations):
                yield event
            return

        plan_ctx.plan = active
        action = ctx.plan_action

        # 2. execute（或 executing 中的后续 turn）→ 注入 plan 走 ReAct，模型用 update_plan 翻状态
        if action == "execute" or (action != "change" and active.status == "executing"):
            if active.status != "executing":
                active.status = "executing"
                await self._plan_repo.save(active)
            messages = self._assemble(ctx)
            messages.append(ChatMessage(session_id=ctx.session_id, role=Role.USER, content=execute_plan_block(active)))
            messages.append(ChatMessage(session_id=ctx.session_id, role=Role.USER, content=EXECUTE_DIRECTIVE))
            async for event in self._drive(ctx, messages, scope, plan_ctx, max_iterations):
                yield event
            if active.steps and all(s.status == "completed" for s in active.steps):
                active.status = "completed"
                await self._plan_repo.save(active)
            return

        # 3. change（含 awaiting_review 下的新消息）→ 比对手改 + 带建议让模型 update_plan 重规划
        manually_edited = active.compute_hash() != active.content_hash
        messages = self._assemble(ctx)
        messages.append(ChatMessage(session_id=ctx.session_id, role=Role.USER, content=change_plan_block(active)))
        messages.append(ChatMessage(
            session_id=ctx.session_id, role=Role.USER,
            content=change_directive(ctx.plan_feedback or rm.user_query, manually_edited),
        ))
        async for event in self._drive(ctx, messages, scope, plan_ctx, max_iterations):
            yield event

    def _assemble(self, ctx: OrchestrationContext) -> List[ChatMessage]:
        rm = ctx.raw_materials
        return ctx.assembler.assemble_prompt(
            session_id=ctx.session_id,
            user_query=rm.user_query,
            system_prompt=rm.system_prompt,
            session_summary=rm.session_summary,
            history_messages=rm.history_messages,
            relevant_facts=rm.relevant_facts,
            frontend_states=rm.frontend_states,
            available_skills=rm.available_skills or None,
        )

    async def _drive(
        self,
        ctx: OrchestrationContext,
        messages: List[ChatMessage],
        scope: ToolScope,
        plan_ctx: PlanContext,
        max_iterations: int,
    ) -> AsyncIterator[StreamEvent]:
        """复用 ReAct 单步循环；每轮 drain plan_ctx 工具事件（PlanCreated/StepStatus/Updated）并 yield"""
        final_message = None
        for iteration in range(max_iterations):
            step_finish = None
            async for item in self._step_runner.run(
                messages=messages,
                session_id=ctx.session_id,
                model_request=ctx.model,
                iteration=iteration,
                tool_scope=scope,
            ):
                if isinstance(item, StepFinishEvent):
                    step_finish = item
                yield item

            for event in plan_ctx.drain_events():
                yield event

            assert step_finish is not None
            ctx.usage_tokens += step_finish.token_usage
            if step_finish.is_finished:
                final_message = step_finish.final_assistant_message
                break
            messages.extend(step_finish.intermediate_messages)
            ctx.record_messages.extend(step_finish.intermediate_messages)

        if final_message is not None:
            ctx.record_messages.append(final_message)
