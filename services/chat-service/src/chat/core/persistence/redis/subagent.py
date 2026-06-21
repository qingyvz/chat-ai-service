import redis.asyncio as redis
from typing import Optional

from chat.application.agents import AgentInfo
from chat.core.config.app_settings import settings


class RedisSubAgentRepository:
    """subagent 的 Redis 存储：spec 存 subagent:{session_id}:{subagent_id}，TTL 跟会话"""

    def __init__(self):
        self.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.ttl = 3600 * 24  # 跟会话热缓存一致，24 小时

    def _spec_key(self, session_id: str, subagent_id: str) -> str:
        return f"subagent:{session_id}:{subagent_id}"

    async def save(self, session_id: str, subagent_id: str, agent: AgentInfo) -> None:
        await self.redis.set(self._spec_key(session_id, subagent_id), agent.model_dump_json(), ex=self.ttl)

    async def get(self, session_id: str, subagent_id: str) -> Optional[AgentInfo]:
        raw = await self.redis.get(self._spec_key(session_id, subagent_id))
        return AgentInfo.model_validate_json(raw) if raw else None
