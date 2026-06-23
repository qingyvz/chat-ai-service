from typing import Optional, Protocol

from chat.application.agents.agent_info import AgentInfo


class SubAgentRepository(Protocol):
    """subagent 临时存储（Redis）：spec 随会话 TTL，纯 py 端创建不走 Java 持久化"""

    async def save(self, session_id: str, subagent_id: str, agent: AgentInfo) -> None:
        ...

    async def get(self, session_id: str, subagent_id: str) -> Optional[AgentInfo]:
        ...
