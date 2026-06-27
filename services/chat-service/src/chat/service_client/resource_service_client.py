from __future__ import annotations

from typing import Mapping, Optional

from common.core.domain import GroupRoleType
from common.core.exceptions import RpcError
from common.http.rpc_client import RpcClient


_DEFAULT_SERVICE_NAME = "resource-service"
_CHECK_RES_PERMISSION_PATH = "/internal/resource/checkResPermission"
_ADD_RES_PATH = "/internal/resource/addRes"


class ResourceClient:
    def __init__(
        self,
        rpc: RpcClient,
        *,
        service_name: str = _DEFAULT_SERVICE_NAME,
    ) -> None:
        self._rpc = rpc
        self._service_name = service_name

    async def check_res_permission(
        self,
        resource_id: str,
        user_id: str | int,
        group_role_map: Mapping[str, GroupRoleType],
    ) -> Optional[dict]:
        resource_id = (resource_id or "").strip()
        try:
            data = await self._rpc.post(
                self._service_name,
                _CHECK_RES_PERMISSION_PATH,
                json={
                    "resourceId": resource_id,
                    "userId": int(user_id),
                    "groupRoles": self._serialize_group_roles(group_role_map),
                },
            )
        except RpcError as e:
            raise e
        if not isinstance(data, dict):
            raise RpcError(
                service_name=self._service_name, path=_CHECK_RES_PERMISSION_PATH,
                msg=f"unexpected data payload: {data!r}",
            )
        return data

    async def create_resource_item(
        self,
        *,
        resource_name: str,
        owner_id: str,
        resource_type: str = "plan",
        path_tag_id: Optional[str] = None,
        preview: Optional[str] = None,
        size: Optional[int] = None,
    ) -> str:
        """内部注册资源（/internal/resource/addRes），返回 resourceId"""
        body: dict = {"resourceName": resource_name, "resourceType": resource_type, "ownerId": owner_id}
        if path_tag_id is not None:
            body["pathTagId"] = path_tag_id
        if preview is not None:
            body["preview"] = preview
        if size is not None:
            body["size"] = size
        data = await self._rpc.post(self._service_name, _ADD_RES_PATH, json=body)
        if not isinstance(data, str) or not data:
            raise RpcError(
                service_name=self._service_name, path=_ADD_RES_PATH,
                msg=f"unexpected data payload: {data!r}",
            )
        return data

    @staticmethod
    def _serialize_group_roles(group_role_map: Mapping[str, GroupRoleType]) -> dict[str, int]:
        serialized: dict[str, int] = {}
        for group_id, role in (group_role_map or {}).items():
            try:
                serialized[str(group_id)] = int(role.code)
            except (AttributeError, TypeError, ValueError):
                continue
        return serialized
