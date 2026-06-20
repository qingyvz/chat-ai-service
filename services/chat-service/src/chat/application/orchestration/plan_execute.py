import json
import uuid
from typing import AsyncIterator, Dict, List, Tuple

from chat.domain.entities import ChatMessage, Role, Plan, PlanStep
from chat.domain.interfaces import LLMProvider
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
    """Plan-and-Execute（v1 线性单角色）：Planner 一次性产出计划 → 逐步隔离执行 → Synthesizer 汇总终答"""

    def __init__(self, llm: LLMProvider, sub_runtime) -> None:
        self._llm = llm
        self._sub_runtime = sub_runtime  # SubAgentTurnRuntime（鸭子类型，避免 runtime↔orchestration 循环导入）

    async def run(self, ctx: OrchestrationContext) -> AsyncIterator[StreamEvent]:
        rm = ctx.raw_materials
        # 记账下放：seed 本轮 user 记录消息（终答在 Synthesizer 末尾补 assistant 消息）
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

        # 1. Planner：一次 LLM 调用产出整张 to-do list
        plan = await self._plan(ctx)
        yield PlanCreatedEvent(plan_id=plan.plan_id, steps=self._steps_payload(plan))

        # 2. 逐步执行（v1 线性）：每步派一个 subagent 子 runtime（多角色分工）
        prior_results: List[Tuple[str, str]] = []
        for step in plan.steps:
            step.status = "in_progress"
            yield PlanStepStatusEvent(step_id=step.step_id, status="in_progress")

            holder: Dict[str, str] = {"text": ""}
            async for ev in self._sub_runtime.run(
                parent_ctx=ctx,
                role="executor",
                step=step,
                prior_results=prior_results,
                holder=holder,
            ):
                yield ev

            step.status = "completed"
            step.result_summary = holder["text"]
            prior_results.append((step.title, holder["text"]))
            yield PlanStepStatusEvent(step_id=step.step_id, status="completed", result_summary=holder["text"])

        # 3. Synthesizer：汇总各步结果 → 终答文本（复用 text 事件）
        async for ev in self._synthesize(ctx, plan):
            yield ev

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

        result = await self._llm.chat_completion(
            messages=messages,
            model_name=ctx.model.model_name,
            temperature=0.2,
            api_base=ctx.model.api_base_url,
            api_key=ctx.model.api_key,
        )
        ctx.usage_tokens += result.usage_tokens
        content = result.raw.choices[0].message.content or ""
        return Plan(plan_id=f"plan_{uuid.uuid4().hex}", steps=self._parse_plan_steps(content, fallback_query=rm.user_query))

    async def _synthesize(self, ctx: OrchestrationContext, plan: Plan) -> AsyncIterator[StreamEvent]:
        """汇总各步结果，流式产出终答文本"""
        synth_messages = [
            ChatMessage(session_id=ctx.session_id, role=Role.SYSTEM, content=_SYNTH_SYSTEM_PROMPT),
            ChatMessage(session_id=ctx.session_id, role=Role.USER, content=self._build_synth_input(ctx.raw_materials.user_query, plan)),
        ]

        text_id = f"txt_{uuid.uuid4().hex}"
        reasoning_id = f"rsn_{uuid.uuid4().hex}"
        interpreter = StepDeltaInterpreter(text_id=text_id, reasoning_id=reasoning_id)

        yield StepStartEvent()
        usage_tokens = 0
        async for chunk in self._llm.stream_chat_completion(
            messages=synth_messages,
            model_name=ctx.model.model_name,
            api_base=ctx.model.api_base_url,
            api_key=ctx.model.api_key,
        ):
            usage_tokens += chunk.usage_tokens
            choices = chunk.raw.choices
            if choices:
                for event in interpreter.consume(choices[0].delta):
                    yield event
        for event in interpreter.close():
            yield event

        ctx.usage_tokens += usage_tokens
        final_message = ChatMessage(
            session_id=ctx.session_id,
            role=Role.ASSISTANT,
            model_id=ctx.model.model_id,
            content=interpreter.assistant_content or "",
            reasoning_content=interpreter.assistant_reasoning or None,
        )
        ctx.record_messages.append(final_message)
        yield StepFinishEvent(is_finished=True, final_assistant_message=final_message, usage_tokens=usage_tokens)

    @staticmethod
    def _steps_payload(plan: Plan) -> List[Dict[str, str]]:
        return [{"id": s.step_id, "title": s.title, "status": s.status} for s in plan.steps]

    @staticmethod
    def _build_synth_input(user_query: str, plan: Plan) -> str:
        results = "\n\n".join([f"[{s.title}]\n{s.result_summary or ''}" for s in plan.steps])
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
        for idx, raw in enumerate(raw_steps):
            title = (raw.get("title") or "").strip()
            if not title:
                continue
            steps.append(PlanStep(
                step_id=f"step_{idx + 1}_{uuid.uuid4().hex[:6]}",
                title=title,
                description=(raw.get("description") or "").strip(),
            ))

        if not steps:
            steps.append(PlanStep(step_id=f"step_1_{uuid.uuid4().hex[:6]}", title="执行任务", description=fallback_query))
        return steps
