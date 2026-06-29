from __future__ import annotations

from typing import List, Optional, Set

from chat.domain.entities import SkillMeta, Skill, Plan, PlanStep
from common.core.exceptions import RpcError
from common.http.rpc_client import RpcClient


_DEFAULT_SERVICE_NAME = "ai-asset-service"
_GET_SKILL_PATH = "/internal/skill/getSkillByResourceId"
_LIST_PUBLISHED_SKILLS_META_PATH = "/internal/skill/listPublishedSkillsMetaByResourceIds"
_CREATE_PLAN_PATH = "/internal/plan/create"
_LIST_PLANS_BY_OWNER_PATH = "/internal/plan/listByOwner"
_UPDATE_PLAN_PATH = "/internal/plan/update"

_ACTIVE_PLAN_STATUSES = ("awaiting_review", "executing")


class AIAssetClient:
    def __init__(
        self,
        rpc: RpcClient,
        *,
        service_name: str = _DEFAULT_SERVICE_NAME,
    ) -> None:
        self._rpc = rpc
        self._service_name = service_name

    async def list_published_skills_meta(self, skill_ids: Set[str]) -> List[SkillMeta]:
        payloads = await self._list_published_skills_meta_by_resource_ids(skill_ids)
        metas = [SkillMeta.from_response(item) for item in payloads]
        return [meta for meta in metas if meta.skill_id]

    async def get_skill_with_version(self, skill_id: str, skill_version: int) -> Optional[Skill]:
        published_skill_res = await self._get_skill_by_resource_id(skill_id, skill_version)
        return Skill.from_response(published_skill_res)

    async def get_published_skill(self, skill_id: str) -> Optional[Skill]:
        published_skill_res = await self._get_skill_by_resource_id(skill_id)
        return Skill.from_response(published_skill_res)

    async def _get_skill_by_resource_id(self, resource_id: str, skill_version: int = None) -> dict:
        try:
            data = await self._rpc.get(
                self._service_name,
                _GET_SKILL_PATH,
                params={"resourceId": resource_id, "skillVersion": skill_version},
            )
        except RpcError as e:
            raise e
        if not isinstance(data, dict):
            raise RpcError(
                service_name=self._service_name, path=_GET_SKILL_PATH,
                msg=f"unexpected data payload: {data!r}",
            )
        return data

    async def create_plan(
        self,
        *,
        owner_id: str,
        title: str,
        content: str,
        steps: List[dict],
        description: str = "",
    ) -> dict:
        """注册计划文件资产（owner 维度、不绑会话；ai-asset 落 Mongo 单档），返回 PlanInfoResponse"""
        data = await self._rpc.post(
            self._service_name,
            _CREATE_PLAN_PATH,
            json={
                "ownerId": owner_id, "title": title,
                "name": title, "description": description, "content": content, "steps": steps,
            },
        )
        if not isinstance(data, dict):
            raise RpcError(
                service_name=self._service_name, path=_CREATE_PLAN_PATH,
                msg=f"unexpected data payload: {data!r}",
            )
        return data

    async def get_active_plan(self, owner_id: str) -> Optional[Plan]:
        """取该用户最近一个未完成的计划：listByOwner 按 updateTime 倒序，筛活跃态取首个（冷读回源，热路径走 Redis）"""
        data = await self._rpc.get(
            self._service_name, _LIST_PLANS_BY_OWNER_PATH, params={"ownerId": owner_id},
        )
        for item in (data or []):
            if (item.get("status") or "").lower() in _ACTIVE_PLAN_STATUSES:
                return self._plan_from_response(item, owner_id)
        return None

    async def update_plan(
        self,
        *,
        resource_id: str,
        content: Optional[str] = None,
        steps: Optional[List[dict]] = None,
        status: Optional[str] = None,
    ) -> None:
        """更新计划内容/步骤/状态（content 变更时 ai-asset 重传 OSS）"""
        body: dict = {"resourceId": resource_id}
        if content is not None:
            body["content"] = content
        if steps is not None:
            body["steps"] = steps
        if status is not None:
            body["status"] = status
        await self._rpc.post(self._service_name, _UPDATE_PLAN_PATH, json=body)

    @staticmethod
    def _plan_from_response(data: dict, owner_id: str) -> Plan:
        steps = [
            PlanStep(
                step_id=s.get("id") or "", title=s.get("title") or "",
                status=(s.get("status") or "pending").lower(),
            )
            for s in (data.get("steps") or [])
        ]
        resource_id = data.get("resourceId")
        return Plan(
            plan_id=resource_id, user_id=owner_id,
            resource_id=resource_id, object_key=data.get("objectKey"),
            file_name=data.get("name") or "", status=(data.get("status") or "awaiting_review").lower(),
            content=data.get("content") or "", steps=steps, version=data.get("version") or 1,
        )

    async def _list_published_skills_meta_by_resource_ids(self, resource_ids: Set[str]) -> List[dict]:
        try:
            data = await self._rpc.post(
                self._service_name,
                _LIST_PUBLISHED_SKILLS_META_PATH,
                json={"resourceIds": sorted(resource_ids)},
            )
        except RpcError as e:
            raise e
        if not isinstance(data, list):
            raise RpcError(
                service_name=self._service_name, path=_LIST_PUBLISHED_SKILLS_META_PATH,
                msg=f"unexpected data payload: {data!r}",
            )
        return data
