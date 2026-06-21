import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

from chat.core.config.app_settings import settings
from chat.domain.entities import ChatMessage, Role
from chat.domain.entities.provider import ProviderType
from chat.domain.error_codes import ChatErrorCode
from chat.domain.interfaces import LLMProvider
from chat.domain.interfaces.llm import LLMEventType, LLMStreamEvent, LLMUsage
from chat.domain.entities.message import ToolCallMessage
from chat.domain.repositories.model_repo import ModelRequestInfo
from common.core.exceptions import ServiceException

from .utils import dump_provider_value, without_none


class AnthropicAdapter(LLMProvider):
    """Anthropic Messages API 适配器"""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC

    def runtime_options_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "json_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "thinking": {
                        "type": "object",
                        "oneOf": [
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["type", "budget_tokens"],
                                "properties": {
                                    "type": {"type": "string", "const": "enabled"},
                                    "budget_tokens": {"type": "integer", "minimum": 1024},
                                },
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["type"],
                                "properties": {"type": {"type": "string", "const": "adaptive"}},
                            },
                        ],
                    },
                },
            },
            "defaults": {"thinking": {"type": "adaptive"}},
        }

    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model_request: ModelRequestInfo,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        if AsyncAnthropic is None:
            raise ServiceException(ChatErrorCode.LLM_GENERATION_FAILED, custom_msg="anthropic SDK 未安装")

        kwargs: dict[str, Any] = {"api_key": model_request.api_key}
        if model_request.base_url:
            kwargs["base_url"] = model_request.base_url
        client = AsyncAnthropic(**kwargs)

        anthropic_messages, anthropic_system_message = self._anthropic_messages_formatter(messages)
        request_kwargs: dict[str, Any] = {
            "model": model_request.model_name,
            "messages": anthropic_messages,
            "max_tokens": model_request.model.max_output_tokens or settings.CTX_DEFAULT_OUTPUT_RESERVE_TOKENS,
            "tools": self._anthropic_tools_formatter(tools),
            **model_request.runtime_options,
        }
        if anthropic_system_message:
            request_kwargs["system"] = anthropic_system_message

        try:
            async with client.messages.stream(**without_none(request_kwargs)) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", "")
                    if event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_type = getattr(delta, "type", "")
                        if delta_type == "text_delta":
                            text_delta = getattr(delta, "text", "")
                            if text_delta:
                                yield LLMStreamEvent(type=LLMEventType.TEXT_DELTA, delta=text_delta)
                        elif delta_type == "thinking_delta":
                            thinking_delta = getattr(delta, "thinking", "")
                            if thinking_delta:
                                yield LLMStreamEvent(type=LLMEventType.REASONING_DELTA, delta=thinking_delta)
                final_message = await stream.get_final_message()
        except Exception as e:
            raise ServiceException(ChatErrorCode.LLM_GENERATION_FAILED, custom_msg=f"Anthropic Provider Error: {e}")

        usage = getattr(final_message, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        if input_tokens or output_tokens:
            yield LLMStreamEvent(type=LLMEventType.USAGE, usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens))

        content_blocks = [dump_provider_value(block) for block in getattr(final_message, "content", [])]

        calls = []
        for block in content_blocks:
            if block.get("type") != "tool_use":
                continue
            calls.append(ToolCallMessage(
                call_id=block.get("id") or f"call_{uuid.uuid4().hex}",
                name=block.get("name") or "",
                arguments=block.get("input") if isinstance(block.get("input"), dict) else {},
            ))
        if calls:
            yield LLMStreamEvent(type=LLMEventType.TOOL_CALLS, tool_calls=calls)
        yield LLMStreamEvent(type=LLMEventType.STATE, provider_payload={"content": content_blocks})

    @staticmethod
    def _anthropic_messages_formatter(messages: List[ChatMessage]) -> tuple[list[dict[str, Any]], str]:
        anthropic_system_message = "\n\n".join(msg.content or "" for msg in messages if msg.role == Role.SYSTEM)

        anthropic_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue
            if msg.role == Role.TOOL:
                anthropic_messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": msg.content or ""}],
                })
                continue
            if msg.role == Role.ASSISTANT and msg.model_info and msg.model_info.provider_type == ProviderType.ANTHROPIC and msg.provider_payload:
                anthropic_messages.append({"role": "assistant", "content": msg.provider_payload["content"]})
                continue
            anthropic_messages.append({
                "role": "assistant" if msg.role == Role.ASSISTANT else "user",
                "content": msg.content or "",
            })
        return anthropic_messages, anthropic_system_message

    @staticmethod
    def _anthropic_tools_formatter(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        result = []
        for tool in tools:
            function = tool.get("function", {})
            result.append({
                "name": function.get("name"),
                "description": function.get("description"),
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            })
        return result
