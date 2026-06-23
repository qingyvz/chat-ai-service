import uuid
from typing import AsyncIterator

from chat.core.config.app_settings import settings
from chat.domain.entities import ChatMessage, Role
from common.logger import warn
from chat.application.events import (
    StepFinishEvent,
    StepStartEvent,
    StreamEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
)
from chat.application.orchestration.base import OrchestrationContext, OrchestrationStrategy
from chat.application.orchestration.step_runner import AgentStepRunner


class ReActStrategy(OrchestrationStrategy):
    """ReAct 编排：assemble_prompt 入口组装 + for 循环逐步委派 AgentStepRunner，行为与重构前等价（回归基准）"""

    def __init__(self, step_runner: AgentStepRunner) -> None:
        self._step_runner = step_runner

    async def run(self, ctx: OrchestrationContext) -> AsyncIterator[StreamEvent]:
        rm = ctx.raw_materials
        # 入口组装：ReAct 全程共用一条 messages
        messages = ctx.assembler.assemble_prompt(
            session_id=ctx.session_id,
            user_query=rm.user_query,
            system_prompt=rm.system_prompt,
            session_summary=rm.session_summary,
            history_messages=rm.history_messages,
            relevant_facts=rm.relevant_facts,
            frontend_states=rm.frontend_states,
            available_skills=rm.available_skills or None,
        )

        # 记账下放：策略自己 seed 本轮 user 记录消息
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
        for iteration in range(max_iterations):
            step_finish_event = None
            async for item in self._step_runner.run(
                messages=messages,
                session_id=ctx.session_id,
                model_request=ctx.model,
                iteration=iteration,
                tool_scope=ctx.tool_scope,
            ):
                # StepFinishEvent 既要记账也要透传（SSE 需要 finish-step 帧）
                if isinstance(item, StepFinishEvent):
                    step_finish_event = item
                yield item

            assert step_finish_event is not None
            ctx.usage_tokens += step_finish_event.usage_tokens
            if step_finish_event.is_finished:
                ctx.record_messages.append(step_finish_event.final_assistant_message)
                return
            else:
                ctx.record_messages.extend(step_finish_event.intermediate_messages)
                messages.extend(step_finish_event.intermediate_messages)
        else:
            # 超出最大迭代次数时兜底
            async for event in self._emit_exhausted_warning(ctx.session_id):
                yield event

    async def _emit_exhausted_warning(self, session_id: str) -> AsyncIterator[StreamEvent]:
        """Agent 循环超出最大迭代次数时的兜底文本输出"""
        warning_text = f"Agent 推理超出最大迭代次数{settings.AGENT_MAX_ITERATIONS}，未能生成最终答案"
        warn("tool calling loop exhausted.", session_id=session_id)
        text_id = f"txt_{uuid.uuid4().hex}"
        yield StepStartEvent()
        yield TextStartEvent(text_id=text_id)
        yield TextDeltaEvent(text_id=text_id, delta=warning_text)
        yield TextEndEvent(text_id=text_id)
        final_message = ChatMessage(
            session_id=session_id,
            role=Role.ASSISTANT,
            content=warning_text,
        )
        yield StepFinishEvent(is_finished=True, final_assistant_message=final_message, usage_tokens=0)
