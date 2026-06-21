import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from chat.domain.entities import ChatMessage, Role
from chat.domain.entities.provider import ProviderType
from chat.domain.error_codes import ChatErrorCode
from chat.domain.interfaces import LLMProvider
from chat.domain.interfaces.llm import LLMEventType, LLMStreamEvent, LLMUsage
from chat.domain.entities.message import ToolCallMessage
from chat.domain.repositories.model_repo import ModelRequestInfo
from common.core.exceptions import ServiceException

from .utils import dump_provider_value, json_object, without_none


class OpenAIAdapter(LLMProvider):
    """OpenAI 官方 Responses API 适配器"""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI

    def runtime_options_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "json_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                    "top_p": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "reasoning": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "effort": {"type": "string", "enum": ["none", "minimal", "low", "medium", "high", "xhigh"]},
                            "summary": {"type": "string", "enum": ["auto", "concise", "detailed"]},
                        },
                    },
                },
            },
            "defaults": {"reasoning": {"effort": "medium", "summary": "auto"}},
        }

    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model_request: ModelRequestInfo,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        if AsyncOpenAI is None:
            raise ServiceException(ChatErrorCode.LLM_GENERATION_FAILED, custom_msg="openai SDK 未安装")

        client_kwargs = {"api_key": model_request.api_key}
        if model_request.base_url:
            client_kwargs["base_url"] = model_request.base_url
        client = AsyncOpenAI(**client_kwargs)

        request_input, instructions, previous_response_id = self._openai_messages_formatter(messages)
        request_kwargs: dict[str, Any] = {
            "model": model_request.model_name,
            "input": request_input,
            "instructions": instructions or None,
            "tools": self._openai_tools_formatter(tools),
            "previous_response_id": previous_response_id,
            "stream": True,
            **model_request.runtime_options,
        }

        output_items: list[dict[str, Any]] = []
        current_item: dict[str, Any] | None = None
        response_id: str | None = None
        try:
            stream = await client.responses.create(**without_none(request_kwargs))
            async for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.created":
                    response = getattr(event, "response", None)
                    response_id = getattr(response, "id", None)
                elif event_type == "response.output_item.added":
                    current_item = dump_provider_value(getattr(event, "item", None)) or {}
                elif event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield LLMStreamEvent(type=LLMEventType.TEXT_DELTA, delta=delta)
                elif event_type in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield LLMStreamEvent(type=LLMEventType.REASONING_DELTA, delta=delta)
                elif event_type == "response.function_call_arguments.delta" and current_item is not None:
                    current_item["arguments"] = (current_item.get("arguments") or "") + (getattr(event, "delta", "") or "")
                elif event_type == "response.output_item.done":
                    item = dump_provider_value(getattr(event, "item", None)) or current_item
                    if item:
                        output_items.append(item)
                    current_item = None
                elif event_type == "response.completed":
                    response = getattr(event, "response", None)
                    response_id = getattr(response, "id", None) or response_id
                    usage = getattr(response, "usage", None)
                    token_usage = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
                    if token_usage:
                        yield LLMStreamEvent(type=LLMEventType.USAGE, usage=LLMUsage(output_tokens=token_usage))
        except Exception as e:
            raise ServiceException(ChatErrorCode.LLM_GENERATION_FAILED, custom_msg=f"OpenAI Responses Error: {e}")

        calls: list[ToolCallMessage] = []
        for item in output_items:
            if item.get("type") != "function_call":
                continue
            calls.append(ToolCallMessage(
                call_id=item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}",
                name=item.get("name") or "",
                arguments=json_object(item.get("arguments") or "{}"),
            ))
        if calls:
            yield LLMStreamEvent(type=LLMEventType.TOOL_CALLS, tool_calls=calls)

        yield LLMStreamEvent(type=LLMEventType.STATE, provider_payload={"output": output_items, "response_id": response_id})

    @staticmethod
    def _openai_messages_formatter(messages: List[ChatMessage]) -> tuple[list[Any], str, str | None]:
        instructions = "\n\n".join(msg.content or "" for msg in messages if msg.role == Role.SYSTEM)

        last_response_index = next((
            i
            for i in range(len(messages) - 1, -1, -1)
            if (
                messages[i].role == Role.ASSISTANT
                and messages[i].model_info
                and messages[i].model_info.provider_type == ProviderType.OPENAI
                and messages[i].provider_payload
                and messages[i].provider_payload.get("response_id")
            )
        ), -1)

        if last_response_index >= 0:
            outputs = []
            for msg in messages[last_response_index + 1:]:
                if msg.role == Role.TOOL:
                    outputs.append({"type": "function_call_output", "call_id": msg.tool_call_id, "output": msg.content or ""})
            if outputs:
                return outputs, instructions, messages[last_response_index].provider_payload['response_id']

        items: list[Any] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue
            if msg.role == Role.ASSISTANT and msg.model_info and msg.model_info.provider_type == ProviderType.OPENAI and msg.provider_payload:
                items.extend(msg.provider_payload["output"])
                continue
            if msg.role == Role.TOOL:
                items.append({"type": "function_call_output", "call_id": msg.tool_call_id, "output": msg.content or ""})
                continue
            role = "assistant" if msg.role == Role.ASSISTANT else "user"
            items.append({"role": role, "content": msg.content or ""})
        return items, instructions, None

    @staticmethod
    def _openai_tools_formatter(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        result = []
        for tool in tools:
            function = tool.get("function", {})
            result.append({
                "type": "function",
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            })
        return result
