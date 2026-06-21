import json
import uuid
from typing import AsyncIterator, Dict, List, Tuple

from chat.domain.entities import ChatMessage, Role, Plan, PlanStep
from chat.domain.entities.message import MessageModelInfo
from chat.domain.error_codes import ChatErrorCode
from chat.domain.interfaces.llm import LLMEventType, TextCompletionProvider
from chat.application.llm_provider_resolver import LLMProviderResolver
from common.core.exceptions import ServiceException
from chat.application.events import (
    PlanCreatedEvent,
    PlanStepStatusEvent,
    StepFinishEvent,
    StepStartEvent,
    StreamEvent,
)
from chat.application.orchestration.base import OrchestrationContext, OrchestrationStrategy
from chat.application.orchestration.delta_interpreter import StepDeltaInterpreter

_PLANNER_DIRECTIVE = (
    "Break the request in <user_query> into an ordered list of concrete, self-contained subtasks "
    "for step-by-step execution.\n"
    "Output STRICT JSON only — no prose, no code fence:\n"
    '{"steps":[{"title":"短标题","description":"可独立执行的子任务说明"}]}\n'
    "Keep the steps minimal; each description must be executable in isolation."
)
_SYNTH_SYSTEM_PROMPT = (
    "You are a synthesis agent. Given the user's original request and the results of each executed subtask, "
    "produce one coherent final answer in the user's language. Do not mention the planning process."
)


class PlanAndExecuteStrategy(OrchestrationStrategy):
    """
    Plan-and-Execute（v1 线性单角色）：Planner 产计划 → 每步调 create_subagent/call_subagent 两个工具执行 → Synthesizer 汇总终答。
    spawn 机制不在策略内，落在两个工具里；策略只负责"产计划 + 发 to-do 事件 + 逐步调工具"。
    """

    def __init__(self, text_provider: TextCompletionProvider, resolver: LLMProviderResolver) -> None:
        self._text_provider = text_provider
        self._resolver = resolver

    async def run(self, ctx: OrchestrationContext) -> AsyncIterator[StreamEvent]:
        raw_materials = ctx.raw_materials
        # 记账下放：seed 本轮 user 记录消息（终答在 Synthesizer 末尾补 assistant 消息）
        ctx.record_messages.append(ChatMessage(
            session_id=ctx.session_id,
            role=Role.USER,
            content=raw_materials.user_query,
            metadata={
                "relevant_facts": raw_materials.relevant_facts,
                "frontend_states": raw_materials.frontend_states or {},
                "available_skills_id": [skill.skill_id for skill in raw_materials.available_skills] or [],
            },
        ))

        # 从 tool_scope 取 subagent 工具（机制在工具里，策略只调用）
        create_subagent = ctx.tool_scope.get("create_subagent")
        call_subagent = ctx.tool_scope.get("call_subagent")
        tool_context = ctx.tool_scope.context
        if create_subagent is None or call_subagent is None:
            raise ServiceException(ChatErrorCode.SUBAGENT_TOOLS_UNAVAILABLE)

        # 1. Planner：一次 LLM 调用产出整张 to-do list
        plan = await self._plan(ctx)
        yield PlanCreatedEvent(plan_id=plan.plan_id, steps=self._steps_payload(plan))

        # 2. 逐步执行（v1 线性）：每步建一个 subagent 并调用它执行子任务
        prior_results: List[Tuple[str, str]] = []
        for step in plan.steps:
            step.status = "in_progress"
            yield PlanStepStatusEvent(step_id=step.step_id, status="in_progress")

            subagent_id = await create_subagent.execute(tool_context, role="executor")
            result_text = await call_subagent.execute(
                tool_context,
                subagent_id=subagent_id,
                task=f"{step.title}\n{step.description}",
                prior_results=prior_results,
            )

            step.status = "completed"
            step.result_summary = result_text
            prior_results.append((step.title, result_text))
            yield PlanStepStatusEvent(step_id=step.step_id, status="completed", result_summary=result_text)

        # 3. Synthesizer：汇总各步结果 → 终答文本（复用 text 事件）
        async for event in self._synthesize(ctx, plan):
            yield event

    async def _plan(self, ctx: OrchestrationContext) -> Plan:
        """Planner 用全量上下文，追加规划指令，单次调用产出结构化计划"""
        raw_materials = ctx.raw_materials
        messages = ctx.assembler.assemble_prompt(
            session_id=ctx.session_id,
            user_query=raw_materials.user_query,
            system_prompt=raw_materials.system_prompt,
            session_summary=raw_materials.session_summary,
            history_messages=raw_materials.history_messages,
            relevant_facts=raw_materials.relevant_facts,
            frontend_states=raw_materials.frontend_states,
            available_skills=raw_materials.available_skills or None,
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
        return Plan(plan_id=f"plan_{uuid.uuid4().hex}", steps=self._parse_plan_steps(content, fallback_query=raw_materials.user_query))

    async def _synthesize(self, ctx: OrchestrationContext, plan: Plan) -> AsyncIterator[StreamEvent]:
        """汇总各步结果，流式产出终答文本"""
        synth_messages = [
            ChatMessage(session_id=ctx.session_id, role=Role.SYSTEM, content=_SYNTH_SYSTEM_PROMPT),
            ChatMessage(session_id=ctx.session_id, role=Role.USER, content=self._build_synth_input(ctx.raw_materials.user_query, plan)),
        ]

        interpreter = StepDeltaInterpreter()

        yield StepStartEvent()
        usage_tokens = 0
        llm_provider = self._resolver.resolve(ctx.model)
        async for provider_event in llm_provider.stream_chat_completion(
            messages=synth_messages,
            model_request=ctx.model,
        ):
            if provider_event.type == LLMEventType.USAGE and provider_event.usage:
                usage_tokens += provider_event.usage.total_tokens
            for event in interpreter.consume(provider_event):
                yield event
        for event in interpreter.close():
            yield event

        ctx.usage_tokens += usage_tokens
        final_message = ChatMessage(
            session_id=ctx.session_id,
            role=Role.ASSISTANT,
            model_id=ctx.model.model_id,
            model_info=MessageModelInfo.from_model_request(ctx.model),
            content=interpreter.assistant_content or "",
            reasoning_content=interpreter.assistant_reasoning or None,
            token_usage=usage_tokens,
        )
        ctx.record_messages.append(final_message)
        yield StepFinishEvent(is_finished=True, final_assistant_message=final_message, usage_tokens=usage_tokens)

    @staticmethod
    def _steps_payload(plan: Plan) -> List[Dict[str, str]]:
        return [{"id": step.step_id, "title": step.title, "status": step.status} for step in plan.steps]

    @staticmethod
    def _build_synth_input(user_query: str, plan: Plan) -> str:
        results = "\n\n".join([f"[{step.title}]\n{step.result_summary or ''}" for step in plan.steps])
        return f"<user_query>\n{user_query}\n</user_query>\n\n<subtask_results>\n{results}\n</subtask_results>"

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
