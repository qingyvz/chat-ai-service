"""
file-storage-service 的 Python 侧 typed facade
Java RemoteStorageService Feign 接口
"""
from __future__ import annotations

import base64
from typing import Optional

from common.core.exceptions import RpcError
from common.http.rpc_client import RpcClient


_DEFAULT_SERVICE_NAME = "file-storage-service"
_GET_DOWNLOAD_URL_PATH = "/internal/storage/getDownloadUrl"
_UPLOAD_CONTENT_PATH = "/internal/storage/uploadContent"
_PLAN_SCENE = "PRIVATE_PLAN"
_DEFAULT_DOWNLOAD_DURATION_SECONDS = 900


class FileStorageClient:
    def __init__(
        self,
        rpc: RpcClient,
        *,
        service_name: str = _DEFAULT_SERVICE_NAME,
    ) -> None:
        self._rpc = rpc
        self._service_name = service_name

    @property
    def service_name(self) -> str:
        return self._service_name

    async def get_download_url(
        self,
        object_key: str,
        duration_seconds: int = _DEFAULT_DOWNLOAD_DURATION_SECONDS,
    ) -> str:
        data = await self._rpc.get(
            self._service_name,
            _GET_DOWNLOAD_URL_PATH,
            params={"objectKey": object_key, "duration": duration_seconds},
        )
        if not isinstance(data, str) or not data:
            raise RpcError(
                service_name=self._service_name, path=_GET_DOWNLOAD_URL_PATH,
                msg=f"unexpected data payload: {data!r}",
            )
        return data

    async def upload_content(
        self,
        content: str,
        *,
        extension: str = "md",
        scene: str = _PLAN_SCENE,
        biz_tag: Optional[str] = None,
    ) -> str:
        """服务端直传文本内容到 OSS（base64），返回 objectKey"""
        body: dict = {
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "extension": extension,
            "scene": scene,
        }
        if biz_tag is not None:
            body["bizTag"] = biz_tag
        data = await self._rpc.post(self._service_name, _UPLOAD_CONTENT_PATH, json=body)
        object_key = data.get("objectKey") if isinstance(data, dict) else None
        if not object_key:
            raise RpcError(
                service_name=self._service_name, path=_UPLOAD_CONTENT_PATH,
                msg=f"unexpected data payload: {data!r}",
            )
        return object_key
