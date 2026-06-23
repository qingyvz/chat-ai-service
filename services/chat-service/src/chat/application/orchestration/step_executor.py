from dataclasses import dataclass
from typing import List, Tuple

from chat.domain.entities import ChatMessage
from chat.domain.repositories.model_repo import ModelRequestInfo
from chat.application.chat_context_assembler import ChatContextAssembler
from chat.application.events import StepFinishEvent
from chat.application.tools import ToolScope
from chat.application.orchestration.step_runner import ReActStepRunner

_STEP_SYSTEM_PROMPT = (
    "You are an execution agent handling ONE isolated subtask of a larger plan. "
    "Use only the provided dependency results and tools; complete the subtask and return a concise result. "
    "Do not plan or mention other subtasks."
)


@dataclass
class StepResult:
    text: str
    tokens: int
    ok: bool


class StepExecutor:
    """单 step 的隔离 executor：用依赖结果造隔离上下文，跑一条 ReAct 内层循环到结束，事件 opaque（不外显）"""

    def __init__(
        self,
        *,
        step_runner: ReActStepRunner,
        assembler: ChatContextAssembler,
        model: ModelRequestInfo,
        session_id: str,
        inner_tool_scope: ToolScope,
        max_iterations: int,
    ) -> None:
        self._step_runner = step_runner
        self._assembler = assembler
        self._model = model
        self._session_id = session_id
        self._inner_tool_scope = inner_tool_scope
        self._max_iterations = max_iterations

    async def run(self, title: str, description: str, dep_results: List[Tuple[str, str]]) -> StepResult:
        prior_facts = [f"{dep_title}: {summary}" for dep_title, summary in dep_results]
        messages = self._assembler.assemble_prompt(
            session_id=self._session_id,
            user_query=f"{title}\n{description}".strip(),
            system_prompt=_STEP_SYSTEM_PROMPT,
            session_summary=None,
            history_messages=[],
            relevant_facts=prior_facts,
        )

        tokens = 0
        for iteration in range(self._max_iterations):
            step = None
            async for item in self._step_runner.run(
                messages=messages,
                session_id=self._session_id,
                model_request=self._model,
                iteration=iteration,
                tool_scope=self._inner_tool_scope,
            ):
                if isinstance(item, StepFinishEvent):
                    step = item
                # 内层事件 opaque：仅取最终结论，不外显
            assert step is not None
            tokens += step.token_usage
            if step.is_finished:
                return StepResult(text=step.final_assistant_message.content or "", tokens=tokens, ok=True)
            messages.extend(step.intermediate_messages)
        return StepResult(text="子任务超出最大迭代次数未能完成", tokens=tokens, ok=False)
