import redis.asyncio as redis
from typing import Optional

from chat.domain.entities import Plan
from chat.core.config.app_settings import settings


class RedisPlanCache:
    """活跃计划的 Redis 热缓存：plan:active:{session_id}，真相源在 ai-asset，TTL 跟会话"""

    def __init__(self):
        self.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.ttl = 3600 * 24

    def _key(self, session_id: str) -> str:
        return f"plan:active:{session_id}"

    async def save(self, session_id: str, plan: Plan) -> None:
        await self.redis.set(self._key(session_id), plan.model_dump_json(), ex=self.ttl)

    async def get(self, session_id: str) -> Optional[Plan]:
        raw = await self.redis.get(self._key(session_id))
        return Plan.model_validate_json(raw) if raw else None

    async def clear(self, session_id: str) -> None:
        await self.redis.delete(self._key(session_id))
