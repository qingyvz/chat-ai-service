from typing import List, Optional, Protocol

from chat.application.agents.agent import Agent
from chat.domain.entities import ChatMessage


class SubAgentRepository(Protocol):
    """subagent 临时存储（Redis）：spec 与 transcript 随会话 TTL，纯 py 端创建不走 Java 持久化"""

    async def save(self, session_id: str, step_id: str, agent: Agent) -> None:
        ...

    async def get(self, session_id: str, step_id: str) -> Optional[Agent]:
        ...

    async def save_transcript(self, session_id: str, step_id: str, messages: List[ChatMessage]) -> None:
        ...

    async def get_transcript(self, session_id: str, step_id: str) -> List[ChatMessage]:
        ...
