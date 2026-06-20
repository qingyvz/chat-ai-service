from dataclasses import dataclass
from typing import List, Optional, Protocol

from chat.application.agents import AgentMemoryPolicy
from chat.application.chat_context_assembler import ChatContextAssembler, WindowedMessages
from chat.domain.entities import ChatMessage
from chat.domain.interfaces.memory import MemoryProvider


@dataclass
class SessionContext:
    """会话记忆原料：历史明细 / 长期记忆事实 / 历史摘要 / 压缩窗口"""
    history_messages: List[ChatMessage]
    relevant_facts: List[str]
    session_summary: Optional[str]
    windowed_history_messages: Optional[WindowedMessages]


class ContextProvider(Protocol):
    """取记忆原料；环境可换：Session 用短期历史+mem0+摘要+窗口，SubAgent 用隔离的子任务+前序产出"""
    async def load(
        self,
        user_id: str,
        session_id: str,
        user_query: str,
        memory_policy: AgentMemoryPolicy,
        prompt_budget_tokens: int,
    ) -> SessionContext:
        ...


class SessionContextProvider:
    """会话场景：只管会话记忆原料的加载（短期历史 / mem0 长期记忆 / 摘要 / 压缩窗口）"""

    def __init__(self, memory: MemoryProvider, assembler: ChatContextAssembler) -> None:
        self._memory = memory
        self._assembler = assembler

    async def load(
        self,
        user_id: str,
        session_id: str,
        user_query: str,
        memory_policy: AgentMemoryPolicy,
        prompt_budget_tokens: int,
    ) -> SessionContext:
        # 加载会话历史 (若启用)：Redis 读取，缓存失效时从 MongoDB 回填
        if memory_policy.enable_chat_memory:
            history_messages = await self._assembler.get_chat_history_record_messages(session_id)
        else:
            history_messages = []

        # 加载长期记忆 (若启用)：按相似度阈值召回跨会话事实
        relevant_facts = []
        if memory_policy.enable_long_term_memory:
            relevant_facts = await self._memory.search(
                user_id=user_id,
                query=user_query,
                limit=memory_policy.long_term_memory_limit,
                score_threshold=memory_policy.long_term_memory_score_threshold,
            )

        # 加载历史摘要与压缩窗口 (若启用，前提是启用会话历史)
        session_summary = None
        windowed_history_messages = None
        if memory_policy.enable_chat_memory and memory_policy.enable_chat_memory_summary:
            session_summary = await self._assembler.get_session_summary(session_id)
            windowed_history_messages = await self._assembler.build_windowed_messages(
                history_messages,
                prompt_budget_tokens=prompt_budget_tokens,
                high_watermark_ratio=memory_policy.high_watermark_ratio,
                low_watermark_ratio=memory_policy.low_watermark_ratio,
            )

        return SessionContext(
            history_messages=history_messages,
            relevant_facts=relevant_facts,
            session_summary=session_summary,
            windowed_history_messages=windowed_history_messages,
        )
