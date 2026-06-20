from typing import Protocol

from chat.application.agents import Agent, AgentResolver, DefaultAgentResolver
from chat.domain.repositories import SessionRepository


class AgentProvider(Protocol):
    """取本轮 Agent spec；环境可换：Session 走 Mongo/default，SubAgent 走 Redis"""
    async def resolve(self, session_id: str, user_id: str) -> Agent | None:
        ...


class SessionAgentProvider:
    """会话场景：按 session 绑定的 agent_id 解析持久化 Agent"""

    def __init__(self, agent_resolver: AgentResolver | None, session_repo: SessionRepository) -> None:
        self._agent_resolver = agent_resolver or DefaultAgentResolver()
        self._session_repo = session_repo

    async def resolve(self, session_id: str, user_id: str) -> Agent | None:
        session = await self._session_repo.get_session_for_user(session_id, user_id)
        return await self._agent_resolver.resolve(session.agent_id)
