import json
import uuid
from typing import AsyncIterator, Dict, List

from chat.core.config.app_settings import settings
from chat.domain.entities import ChatMessage, Role, Plan, PlanStep
from chat.domain.interfaces.llm import TextCompletionProvider
from common.logger import warn
from chat.application.events import (
    PlanCreatedEvent,
    PlanStepStatusEvent,
    StepFinishEvent,
    StepStartEvent,
    StreamEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolInputAvailableEvent,
)
from chat.application.orchestration.base import OrchestrationContext, OrchestrationStrategy
from chat.application.orchestration.step_runner import PlanExecuteStepRunner

_COMPLETE_STEP_TOOL = "complete_plan_step"

_PLANNER_DIRECTIVE = (
    "Break the request in <user_query> into an ordered list of concrete, self-contained subtasks "
    "for step-by-step execution.\n"
    "Output STRICT JSON only — no prose, no code fence:\n"
    '{"steps":[{"title":"短标题","description":"可独立执行的子任务说明"}]}\n'
    "Keep the steps minimal; each description must be executable in isolation."
)


class PlanAndExecuteStrategy(OrchestrationStrategy):
    """Plan-and-Execute（v1 线性）：Planner 产 to-do list 注入上下文 → 复用 ReAct 内核让 model 按 list 自驱执行（是否调 subagent 由 model 决定，不在编排里硬编码）"""

    def __init__(self, step_runner: PlanExecuteStepRunner, text_provider: TextCompletionProvider) -> None:
        self._step_runner = step_runner
        self._text_provider = text_provider

    async def run(self, ctx: OrchestrationContext) -> AsyncIterator[StreamEvent]:
        rm = ctx.raw_materials
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

        # 1. Planner：一次 LLM 调用产出整张 to-do list（结构化 JSON）
        plan = await self._plan(ctx)
        yield PlanCreatedEvent(plan_id=plan.plan_id, steps=self._steps_payload(plan))
        steps_by_id = {step.step_id: step for step in plan.steps}

        # 2. 把计划注入上下文，复用 ReAct 内核让 model 按 list 自驱（工具/subagent 调用交给 model 决定）
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
        messages.append(ChatMessage(session_id=ctx.session_id, role=Role.USER, content=self._execution_directive(plan)))

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
                if isinstance(item, StepFinishEvent):
                    step_finish_event = item
                # 拦 model 的进度上报：complete_plan_step → 翻译成领域 PlanStepStatusEvent（事件翻译留在编排层）
                elif isinstance(item, ToolInputAvailableEvent) and item.tool_name == _COMPLETE_STEP_TOOL:
                    yield self._step_completed_event(steps_by_id, item.input)
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
            async for event in self._emit_exhausted_warning(ctx.session_id):
                yield event

    async def _plan(self, ctx: OrchestrationContext) -> Plan:
        """Planner 用全量上下文，追加规划指令，单次调用产出结构化计划"""
        rm = ctx.raw_materials
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
        messages.append(ChatMessage(session_id=ctx.session_id, role=Role.USER, content=_PLANNER_DIRECTIVE))

        result = await self._text_provider.chat_completion(
            messages=messages,
            model_name=ctx.model.model_name,
            temperature=0.2,
            api_base=ctx.model.api_base_url,
            api_key=ctx.model.api_key,
        )
        ctx.usage_tokens += result.usage_tokens
        content = result.raw.choices[0].message.content or ""
        return Plan(plan_id=f"plan_{uuid.uuid4().hex}", steps=self._parse_plan_steps(content, fallback_query=rm.user_query))

    async def _emit_exhausted_warning(self, session_id: str) -> AsyncIterator[StreamEvent]:
        """执行循环超出最大迭代次数时的兜底文本输出"""
        warning_text = f"Plan-and-Execute 推理超出最大迭代次数{settings.AGENT_MAX_ITERATIONS}，未能生成最终答案"
        warn("plan-execute loop exhausted.", session_id=session_id)
        text_id = f"txt_{uuid.uuid4().hex}"
        yield StepStartEvent()
        yield TextStartEvent(text_id=text_id)
        yield TextDeltaEvent(text_id=text_id, delta=warning_text)
        yield TextEndEvent(text_id=text_id)
        final_message = ChatMessage(session_id=session_id, role=Role.ASSISTANT, content=warning_text)
        yield StepFinishEvent(is_finished=True, final_assistant_message=final_message, usage_tokens=0)

    @staticmethod
    def _execution_directive(plan: Plan) -> str:
        """把 to-do list 渲染成执行指令注入上下文"""
        todo = "\n".join([f"{i + 1}. [{s.step_id}] {s.title}: {s.description}" for i, s in enumerate(plan.steps)])
        return (
            "You have produced the following plan. Execute the steps in order to fulfill the user's request.\n"
            f"Plan:\n{todo}\n"
            "After finishing each step, call complete_plan_step(step_id, summary) to report progress. "
            "You MAY use create_subagent/call_subagent to run a step in isolation if it helps, but it is optional. "
            "When all steps are done, produce the final answer in the user's language."
        )

    @staticmethod
    def _step_completed_event(steps_by_id: Dict[str, PlanStep], tool_input: Dict[str, str]) -> PlanStepStatusEvent:
        """据 model 上报的 complete_plan_step 入参产出步状态事件，并回写计划内的步状态"""
        step_id = (tool_input.get("step_id") or "").strip()
        summary = tool_input.get("summary")
        step = steps_by_id.get(step_id)
        if step is not None:
            step.status = "completed"
            step.result_summary = summary
        return PlanStepStatusEvent(step_id=step_id, status="completed", result_summary=summary)

    @staticmethod
    def _steps_payload(plan: Plan) -> List[Dict[str, str]]:
        return [{"id": step.step_id, "title": step.title, "status": step.status} for step in plan.steps]

    @staticmethod
    def _parse_plan_steps(content: str, fallback_query: str) -> List[PlanStep]:
        """解析 Planner 的 JSON 输出；失败则兜底为覆盖原始请求的单步计划"""
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text[:4].lower() == "json":
                text = text[4:]
        try:
            raw_steps = json.loads(text).get("steps", [])
        except (json.JSONDecodeError, AttributeError):
            raw_steps = []

        steps: List[PlanStep] = []
        for index, raw in enumerate(raw_steps):
            title = (raw.get("title") or "").strip()
            if not title:
                continue
            steps.append(PlanStep(
                step_id=f"step_{index + 1}_{uuid.uuid4().hex[:6]}",
                title=title,
                description=(raw.get("description") or "").strip(),
            ))

        if not steps:
            steps.append(PlanStep(step_id=f"step_1_{uuid.uuid4().hex[:6]}", title="执行任务", description=fallback_query))
        return steps
