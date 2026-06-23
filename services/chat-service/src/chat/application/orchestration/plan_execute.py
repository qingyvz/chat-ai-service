import json
import uuid
from typing import AsyncIterator, List

from chat.core.config.app_settings import settings
from chat.domain.entities import ChatMessage, Role, Plan, PlanStep
from chat.domain.interfaces.llm import TextCompletionProvider
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
from chat.application.orchestration.plan_session import PlanSession
from chat.application.orchestration.step_executor import StepExecutor
from chat.application.orchestration.step_runner import PlanExecuteStepRunner

_PLAN_TOOL_NAMES = ("execute_step", "update_plan", "abandon_step")

_PLANNER_DIRECTIVE = (
    "Break the request in <user_query> into a DAG of concrete, self-contained subtasks.\n"
    "Output STRICT JSON only — no prose, no code fence:\n"
    '{"steps":[{"id":"s1","title":"短标题","description":"可独立执行的子任务说明","depends_on":[]}]}\n'
    "Assign each step a short unique id; reference prerequisite ids in depends_on. "
    "Steps with no shared dependency may run in parallel. Keep the steps minimal."
)


class PlanAndExecuteStrategy(OrchestrationStrategy):
    """Plan-and-Execute（DAG，工具驱动 + 重规划自动机）：策略只强制首轮出 plan，之后 model 用 execute_step/update_plan/abandon_step 自驱；并行来自一轮同发多个 execute_step。"""

    def __init__(self, step_runner: PlanExecuteStepRunner, text_provider: TextCompletionProvider) -> None:
        self._step_runner = step_runner
        self._text_provider = text_provider

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

        # 1. 强制首轮出 DAG plan，建 PlanSession + 隔离 executor，绑进 tool_scope
        plan = await self._plan(ctx)
        session = PlanSession(plan, settings.PLAN_EXECUTE_MAX_REPLAN_ATTEMPTS, settings.PLAN_EXECUTE_MAX_PARALLEL)
        session.executor = StepExecutor(
            step_runner=self._step_runner,
            assembler=ctx.assembler,
            model=ctx.model,
            session_id=ctx.session_id,
            inner_tool_scope=ctx.tool_scope.without(*_PLAN_TOOL_NAMES),
            max_iterations=max_iterations,
        )
        exec_scope = ctx.tool_scope.bind("plan_session", session)
        yield PlanCreatedEvent(plan_id=plan.plan_id, steps=session.steps_payload())

        # 2. 注入执行指令
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
        messages.append(ChatMessage(session_id=ctx.session_id, role=Role.USER, content=self._dag_directive(plan)))

        # 3. 自动机外层循环：透传事件 + 拦 in_progress + drain plan 事件 + 失败重规划
        final_message = None
        synth_prompted = False
        for turn in range(max_iterations):
            for failed_id, error in session.take_replans():
                messages.append(ChatMessage(session_id=ctx.session_id, role=Role.USER, content=self._replan_directive(failed_id, error)))

            step_finish = None
            async for item in self._step_runner.run(
                messages=messages,
                session_id=ctx.session_id,
                model_request=ctx.model,
                iteration=turn,
                tool_scope=exec_scope,
            ):
                if isinstance(item, StepFinishEvent):
                    step_finish = item
                elif isinstance(item, ToolInputAvailableEvent) and item.tool_name == "execute_step":
                    step_id = (item.input or {}).get("step_id")
                    step = session.get_step(step_id) if step_id else None
                    if step is not None and session.is_ready(step):
                        yield PlanStepStatusEvent(step_id=step_id, status="in_progress")
                yield item

            # tools 推的 completed/failed/updated 事件
            for event in session.drain_events():
                yield event

            assert step_finish is not None
            ctx.usage_tokens += step_finish.token_usage
            if step_finish.is_finished:
                final_message = step_finish.final_assistant_message
                break
            messages.extend(step_finish.intermediate_messages)
            ctx.record_messages.extend(step_finish.intermediate_messages)

            # 所有步已处理（含被弃）但 model 未收尾 → 提示产出最终答案
            if session.all_done() and not synth_prompted:
                messages.append(ChatMessage(session_id=ctx.session_id, role=Role.USER, content=self._synth_directive()))
                synth_prompted = True

        ctx.usage_tokens += session.step_tokens
        if final_message is not None:
            ctx.record_messages.append(final_message)
            return
        # 兜底：循环耗尽仍未收尾 → 据已完成结果产出部分答案
        async for event in self._forced_finalize(ctx, session):
            yield event

    async def _plan(self, ctx: OrchestrationContext) -> Plan:
        """Planner 用全量上下文，追加规划指令，单次调用产出结构化 DAG 计划"""
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

    async def _forced_finalize(self, ctx: OrchestrationContext, session: PlanSession) -> AsyncIterator[StreamEvent]:
        """循环耗尽兜底：据已完成步结果拼一个部分答案，避免无终答"""
        completed = [s for s in session.plan.steps if s.status == "completed"]
        if completed:
            body = "\n\n".join([f"[{s.title}]\n{s.result_summary or ''}" for s in completed])
            text = f"已完成部分子任务，结果如下：\n\n{body}"
        else:
            text = f"Plan-and-Execute 超出最大迭代次数{settings.AGENT_MAX_ITERATIONS}，未能完成任务。"
        text_id = f"txt_{uuid.uuid4().hex}"
        yield StepStartEvent()
        yield TextStartEvent(text_id=text_id)
        yield TextDeltaEvent(text_id=text_id, delta=text)
        yield TextEndEvent(text_id=text_id)
        final_message = ChatMessage(session_id=ctx.session_id, role=Role.ASSISTANT, content=text)
        ctx.record_messages.append(final_message)
        yield StepFinishEvent(is_finished=True, final_assistant_message=final_message, token_usage=0)

    @staticmethod
    def _dag_directive(plan: Plan) -> str:
        lines = [f"- {s.step_id} (depends_on: {s.depends_on or '无'}): {s.title} — {s.description}" for s in plan.steps]
        return (
            "You have produced the following plan (DAG). Drive it to completion:\n"
            + "\n".join(lines)
            + "\n\nCall execute_step(step_id) for steps whose dependencies are all completed; "
            "issue several execute_step calls in ONE turn for independent ready steps to run them in parallel. "
            "If a step fails, use update_plan to revise/fix it, or abandon_step if it is truly unfixable. "
            "When all reachable steps are done, write the final answer for the user."
        )

    @staticmethod
    def _replan_directive(step_id: str, error: str) -> str:
        return (
            f"Step {step_id} failed: {error}\n"
            "Revise the plan with update_plan to fix it (e.g. split or change approach), then re-run the affected steps. "
            "If it cannot be fixed, call abandon_step to skip it and continue with the rest."
        )

    @staticmethod
    def _synth_directive() -> str:
        return (
            "All reachable steps are now completed or abandoned. "
            "Write the final answer for the user from the completed results, noting any parts that could not be done."
        )

    @staticmethod
    def _parse_plan_steps(content: str, fallback_query: str) -> List[PlanStep]:
        """解析 Planner 的 JSON 输出（含 id / depends_on）；失败则兜底为覆盖原始请求的单步计划"""
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
        seen: set[str] = set()
        for index, raw in enumerate(raw_steps):
            title = (raw.get("title") or "").strip()
            if not title:
                continue
            step_id = (raw.get("id") or "").strip() or f"step_{index + 1}_{uuid.uuid4().hex[:6]}"
            while step_id in seen:
                step_id = f"{step_id}_{uuid.uuid4().hex[:4]}"
            seen.add(step_id)
            steps.append(PlanStep(
                step_id=step_id,
                title=title,
                description=(raw.get("description") or "").strip(),
                depends_on=[str(d).strip() for d in (raw.get("depends_on") or []) if str(d).strip()],
            ))

        if not steps:
            steps.append(PlanStep(step_id=f"step_1_{uuid.uuid4().hex[:6]}", title="执行任务", description=fallback_query))
        # 丢弃指向不存在步的依赖，避免永远 not-ready
        valid_ids = {s.step_id for s in steps}
        for step in steps:
            step.depends_on = [d for d in step.depends_on if d in valid_ids and d != step.step_id]
        return steps
