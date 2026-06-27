from typing import AsyncIterator, List, Optional

from common.logger import warn
from common.core.exceptions import RpcError
from chat.core.config.app_settings import settings
from chat.core.persistence import RedisPlanCache
from chat.domain.entities import ChatMessage, Role, Plan
from chat.service_client import AIAssetClient
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

    def __init__(self, step_runner: ReActStepRunner, ai_asset_client: AIAssetClient, plan_cache: RedisPlanCache) -> None:
        self._step_runner = step_runner
        self._ai_asset_client = ai_asset_client
        self._plan_cache = plan_cache

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
        active = await self._load_active(ctx.session_id, ctx.user_id)

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
                await self._persist_status(active, "executing")
            messages = self._assemble(ctx)
            messages.append(ChatMessage(session_id=ctx.session_id, role=Role.USER, content=execute_plan_block(active)))
            messages.append(ChatMessage(session_id=ctx.session_id, role=Role.USER, content=EXECUTE_DIRECTIVE))
            async for event in self._drive(ctx, messages, scope, plan_ctx, max_iterations):
                yield event
            if active.steps and all(s.status == "completed" for s in active.steps):
                active.status = "completed"
                await self._persist_status(active, "completed")
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

    async def _load_active(self, session_id: str, user_id: str) -> Optional[Plan]:
        """活跃计划：先读 Redis 热缓存，未命中回源 ai-asset 并回填缓存"""
        cached = await self._plan_cache.get(session_id)
        if cached is not None:
            return cached
        active = await self._ai_asset_client.get_active_plan(session_id, user_id)
        if active is not None:
            await self._plan_cache.save(session_id, active)
        return active

    async def _persist_status(self, plan: Plan, status: str) -> None:
        """计划级状态流转：回写 ai-asset + 刷新热缓存"""
        if plan.resource_id:
            try:
                await self._ai_asset_client.update_plan(resource_id=plan.resource_id, status=status)
            except RpcError as e:
                warn("plan status persist failed.", session_id=plan.session_id, detail=str(e))
        await self._plan_cache.save(plan.session_id, plan)

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
