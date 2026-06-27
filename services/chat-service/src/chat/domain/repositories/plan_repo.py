from abc import ABC, abstractmethod
from typing import Optional

from chat.domain.entities import Plan


class PlanRepository(ABC):
    """PlanMode 计划仓储接口 (MongoDB)"""

    @abstractmethod
    async def create(self, plan: Plan) -> Plan: pass

    @abstractmethod
    async def get_active_for_session(self, session_id: str, user_id: str) -> Optional[Plan]: pass

    @abstractmethod
    async def get_by_plan_id(self, plan_id: str, user_id: str) -> Optional[Plan]: pass

    @abstractmethod
    async def save(self, plan: Plan) -> Plan: pass
