from datetime import datetime, timezone
from typing import Optional

from beanie.operators import In

from chat.domain.repositories import PlanRepository
from chat.domain.entities import Plan

_ACTIVE_STATUSES = ["awaiting_review", "executing"]


class MongoPlanRepository(PlanRepository):

    async def create(self, plan: Plan) -> Plan:
        await plan.insert()
        return plan

    async def get_active_for_session(self, session_id: str, user_id: str) -> Optional[Plan]:
        """取该会话当前未完成的计划（awaiting_review / executing），多条取最近一条"""
        return await Plan.find(
            Plan.session_id == session_id,
            Plan.user_id == user_id,
            In(Plan.status, _ACTIVE_STATUSES),
        ).sort("-updated_at").first_or_none()

    async def get_by_plan_id(self, plan_id: str, user_id: str) -> Optional[Plan]:
        return await Plan.find_one(Plan.plan_id == plan_id, Plan.user_id == user_id)

    async def save(self, plan: Plan) -> Plan:
        plan.updated_at = datetime.now(timezone.utc)
        await plan.save()
        return plan
