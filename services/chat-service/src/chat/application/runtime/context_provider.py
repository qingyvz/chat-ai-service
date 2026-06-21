from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple

from common.logger import error, warn
from chat.core.config.app_settings import settings
from chat.application.agents import AgentMemoryPolicy
from chat.domain.entities import ChatMessage, ChatSession
from chat.domain.interfaces.memory import MemoryProvider
from chat.domain.repositories import MessageRepository, HotContextRepository, SessionRepository


@dataclass
class WindowedMessages:
    messages_keep: List[ChatMessage] = field(default_factory=list)
    messages_compress_candidates: List[ChatMessage] = field(default_factory=list)
    needs_compression: bool = False

    def get_messages(self) -> List[ChatMessage]:
        return self.messages_compress_candidates + self.messages_keep


@dataclass
class SessionContext:
    """会话记忆原料：历史明细 / 长期记忆事实 / 历史摘要 / 压缩窗口"""
    history_messages: List[ChatMessage]
    relevant_facts: List[str]
    session_summary: Optional[str]
    windowed_history_messages: Optional[WindowedMessages]


class ContextProvider(Protocol):
    """取上下文原料；环境可换：Session 用短期历史+mem0+摘要+窗口，SubAgent 用隔离的子任务+前序产出"""
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
    """会话场景：短期历史（Redis 读取/降级回填）+ mem0 长期记忆 + 摘要 + 压缩窗口的加载与裁剪"""

    def __init__(
        self,
        memory: MemoryProvider,
        message_repo: MessageRepository,
        session_repo: SessionRepository,
        hot_context_repo: HotContextRepository,
    ) -> None:
        self._memory = memory
        self._message_repo = message_repo
        self._session_repo = session_repo
        self._hot_context_repo = hot_context_repo

    async def load(
        self,
        user_id: str,
        session_id: str,
        user_query: str,
        memory_policy: AgentMemoryPolicy,
        prompt_budget_tokens: int,
    ) -> SessionContext:
        # 加载会话历史 (若启用)
        if memory_policy.enable_chat_memory:
            history_messages = await self._get_chat_history_record_messages(session_id)
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
            session_summary = await self._get_session_summary(session_id)
            windowed_history_messages = await self._build_windowed_messages(
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

    async def _get_chat_history_record_messages(self, session_id: str) -> List[ChatMessage]:
        """从 Redis 拉取短期上下文；缓存失效时从 MongoDB 回填最近 N 条（摘要时间戳之后的未压缩明细）"""
        try:
            recent_messages = await self._hot_context_repo.get_recent_context(session_id)
        except Exception as exc:
            warn("get chat history record messages from read redis hot-context failed.", session_id=session_id, exc=exc)
            recent_messages = []

        if not recent_messages:
            try:
                session: Optional[ChatSession] = await self._session_repo.get_session(session_id)
                history = await self._message_repo.list_session_messages(
                    session_id=session_id,
                    after=session.summary_updated_at,
                    limit=settings.CTX_FALLBACK_HISTORY_LIMIT,
                )
                if history:
                    await self._hot_context_repo.load_messages(session_id, history)
                    return history
            except Exception as exc:
                error("chat history record messages repopulate failed.", session_id=session_id, exc=exc)

        return recent_messages

    async def _get_session_summary(self, session_id: str) -> Optional[str]:
        """从 MongoDB 读取当前会话的摘要（如有）"""
        try:
            session: Optional[ChatSession] = await self._session_repo.get_session(session_id)
            return session.current_summary if session else None
        except Exception:
            return None

    async def _build_windowed_messages(
        self,
        chat_history_record_messages: List[ChatMessage],
        prompt_budget_tokens: int,
        high_watermark_ratio: Optional[float] = None,
        low_watermark_ratio: Optional[float] = None,
    ) -> WindowedMessages:
        """从后往前累加 Token，构建不超过高水位预算的动态滑动窗口；超过高水位则触发摘要"""
        high_ratio = high_watermark_ratio or settings.CTX_HIGH_WATERMARK_RATIO
        low_ratio = low_watermark_ratio or settings.CTX_LOW_WATERMARK_RATIO
        high_budget = int(prompt_budget_tokens * high_ratio)
        low_budget = int(prompt_budget_tokens * low_ratio)

        total_token = 0
        windowed_messages = WindowedMessages()
        for message in reversed(chat_history_record_messages):
            total_token += message.token_count or 0
            if total_token <= low_budget:
                windowed_messages.messages_keep.insert(0, message)
            else:
                windowed_messages.messages_compress_candidates.insert(0, message)

        windowed_messages.needs_compression = total_token >= high_budget
        return windowed_messages


class SubAgentContextProvider:
    """子任务场景：隔离上下文——无会话历史/摘要，前序步骤结论作为 relevant_facts 注入"""

    def __init__(self, prior_results: List[Tuple[str, str]]) -> None:
        self._prior_results = prior_results

    async def load(
        self,
        user_id: str,
        session_id: str,
        user_query: str,
        memory_policy: AgentMemoryPolicy,
        prompt_budget_tokens: int,
    ) -> SessionContext:
        prior_facts = [f"{title}: {summary}" for title, summary in self._prior_results]
        return SessionContext(
            history_messages=[],
            relevant_facts=prior_facts,
            session_summary=None,
            windowed_history_messages=None,
        )
