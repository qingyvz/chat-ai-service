from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, List, Dict, Optional, Any, TYPE_CHECKING
from chat.domain.entities import ChatMessage
from chat.domain.entities.message import ToolCallMessage

if TYPE_CHECKING:
    from chat.domain.entities.provider import ProviderType
    from chat.domain.repositories.model_repo import ModelRequestInfo

@dataclass
class LLMStreamChunk:
    raw: Any
    usage_tokens: int = 0

@dataclass
class LLMCompletionResult:
    raw: Any
    usage_tokens: int


# === provider 归一化（照搬 WisePen-ref）：厂商无关的流式事件 ===

@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMEventType(str, Enum):
    TEXT_DELTA = "TEXT_DELTA"
    REASONING_DELTA = "REASONING_DELTA"
    TOOL_CALLS = "TOOL_CALLS"
    USAGE = "USAGE"
    STATE = "STATE"


@dataclass
class LLMStreamEvent:
    type: LLMEventType
    delta: str | None = None
    tool_calls: list[ToolCallMessage] | None = None
    usage: LLMUsage | None = None
    provider_payload: dict[str, Any] | None = None
    response_id: str | None = None


class TextCompletionProvider(ABC):
    """非流式补全（摘要/标题等轻量调用），从 LLMProvider 拆出"""
    @abstractmethod
    async def chat_completion(
            self,
            messages: List[ChatMessage],
            model_name: str,
            temperature: float = 0.7,
            tools: Optional[List[Dict[str, Any]]] = None,
            api_base: Optional[str] = None,
            api_key: Optional[str] = None,
    ) -> LLMCompletionResult:
        pass


class LLMProvider(ABC):
    """流式补全（归一化为 LLMStreamEvent）；按 provider_type 派发到具体 adapter"""

    @staticmethod
    def empty_runtime_options_manifest() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "json_schema": {"type": "object", "additionalProperties": False, "properties": {}},
            "defaults": {},
        }

    @property
    @abstractmethod
    def provider_type(self) -> "ProviderType":
        pass

    def supports_tools(self) -> bool:
        return True

    def runtime_options_manifest(self) -> dict[str, Any]:
        return self.empty_runtime_options_manifest()

    @abstractmethod
    async def stream_chat_completion(
            self,
            messages: List[ChatMessage],
            model_request: "ModelRequestInfo",
            tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        yield  # type: ignore[misc]
