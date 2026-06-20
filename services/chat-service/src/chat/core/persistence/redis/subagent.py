import json
import redis.asyncio as redis
from typing import List, Optional

from chat.application.agents import Agent
from chat.domain.entities import ChatMessage
from chat.core.config.app_settings import settings


class RedisSubAgentRepository:
    """subagent 的 Redis 存储：spec 存 subagent:{session_id}:{step_id}，transcript 存 :transcript，TTL 跟会话"""

    def __init__(self):
        self.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.ttl = 3600 * 24  # 跟会话热缓存一致，24 小时

    def _spec_key(self, session_id: str, step_id: str) -> str:
        return f"subagent:{session_id}:{step_id}"

    def _transcript_key(self, session_id: str, step_id: str) -> str:
        return f"subagent:{session_id}:{step_id}:transcript"

    async def save(self, session_id: str, step_id: str, agent: Agent) -> None:
        await self.redis.set(self._spec_key(session_id, step_id), agent.model_dump_json(), ex=self.ttl)

    async def get(self, session_id: str, step_id: str) -> Optional[Agent]:
        raw = await self.redis.get(self._spec_key(session_id, step_id))
        return Agent.model_validate_json(raw) if raw else None

    async def save_transcript(self, session_id: str, step_id: str, messages: List[ChatMessage]) -> None:
        if not messages:
            return
        key = self._transcript_key(session_id, step_id)
        serialized = [json.dumps(m.model_dump(mode="json", exclude={"id"}), ensure_ascii=False) for m in messages]
        async with self.redis.pipeline(transaction=True) as pipe:
            await pipe.delete(key)
            await pipe.rpush(key, *serialized)
            await pipe.expire(key, self.ttl)
            await pipe.execute()

    async def get_transcript(self, session_id: str, step_id: str) -> List[ChatMessage]:
        raw_msgs = await self.redis.lrange(self._transcript_key(session_id, step_id), 0, -1)
        return [ChatMessage(**json.loads(m)) for m in raw_msgs]
