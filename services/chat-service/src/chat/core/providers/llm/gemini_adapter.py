import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from chat.domain.entities import ChatMessage, Role
from chat.domain.entities.provider import ProviderType
from chat.domain.error_codes import ChatErrorCode
from chat.domain.interfaces import LLMProvider
from chat.domain.interfaces.llm import LLMEventType, LLMStreamEvent, LLMUsage
from chat.domain.entities.message import ToolCallMessage
from chat.domain.repositories.model_repo import ModelRequestInfo
from common.core.exceptions import ServiceException

from .utils import dump_provider_value, read_provider_value


class GeminiAdapter(LLMProvider):
    """Gemini 官方 Google GenAI SDK 适配器"""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.GOOGLE

    def runtime_options_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "json_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                    "top_p": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "top_k": {"type": "integer", "minimum": 0},
                    "seed": {"type": "integer"},
                    "presence_penalty": {"type": "number"},
                    "frequency_penalty": {"type": "number"},
                    "thinking_config": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "thinking_budget": {"type": "integer"},
                            "thinking_level": {"type": "string", "enum": ["MINIMAL", "LOW", "MEDIUM", "HIGH"]},
                        },
                    },
                },
            },
            "defaults": {"temperature": 0.7, "thinking_config": {"thinking_budget": -1}},
        }

    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model_request: ModelRequestInfo,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        if genai is None:
            raise ServiceException(ChatErrorCode.LLM_GENERATION_FAILED, custom_msg="google-genai SDK 未安装")

        client_kwargs: dict[str, Any] = {"api_key": model_request.api_key}
        if model_request.base_url:
            client_kwargs["http_options"] = types.HttpOptions(base_url=model_request.base_url)
        client = genai.Client(**client_kwargs)

        contents = self._gemini_contents_formatter(messages)
        config_kwargs: dict[str, Any] = {"tools": self._gemini_tools_formatter(tools), **model_request.runtime_options}

        accumulated_parts: list[Any] = []
        final_usage: Any = None
        try:
            stream = await client.aio.models.generate_content_stream(
                model=model_request.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**{key: value for key, value in config_kwargs.items() if value is not None}),
            )
            async for chunk in stream:
                content = self._gemini_get_first_content(chunk)
                if content is not None:
                    parts = read_provider_value(content, "parts", []) or []
                    accumulated_parts.extend(dump_provider_value(part) for part in parts)
                    for part in parts:
                        text = read_provider_value(part, "text")
                        if text:
                            if read_provider_value(part, "thought", False):
                                yield LLMStreamEvent(type=LLMEventType.REASONING_DELTA, delta=text)
                            else:
                                yield LLMStreamEvent(type=LLMEventType.TEXT_DELTA, delta=text)

                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    final_usage = usage
        except Exception as e:
            raise ServiceException(ChatErrorCode.LLM_GENERATION_FAILED, custom_msg=f"Gemini Provider Error: {e}")

        token_usage = int(getattr(final_usage, "total_token_count", 0) or 0) if final_usage else 0
        if token_usage:
            yield LLMStreamEvent(type=LLMEventType.USAGE, usage=LLMUsage(output_tokens=token_usage))

        calls: list[ToolCallMessage] = []
        for part in accumulated_parts:
            function_call = read_provider_value(part, "function_call")
            if function_call:
                name = read_provider_value(function_call, "name", "")
                args = read_provider_value(function_call, "args", {}) or {}
                calls.append(ToolCallMessage(
                    call_id=read_provider_value(function_call, "id") or f"call_{uuid.uuid4().hex}",
                    name=name,
                    arguments=args if isinstance(args, dict) else {},
                ))
        if calls:
            yield LLMStreamEvent(type=LLMEventType.TOOL_CALLS, tool_calls=calls)
        yield LLMStreamEvent(type=LLMEventType.STATE, provider_payload={"content": accumulated_parts})

    @staticmethod
    def _gemini_contents_formatter(messages: List[ChatMessage]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                contents.append({"role": "user", "parts": [{"text": msg.content or ""}]})
                continue
            if msg.role == Role.ASSISTANT and msg.model_info and msg.model_info.provider_type == ProviderType.GOOGLE and msg.provider_payload:
                contents.append({"role": "model", "parts": msg.provider_payload["content"]})
                continue
            if msg.role == Role.TOOL:
                contents.append({
                    "role": "user",
                    "parts": [{"function_response": {"name": msg.name or "", "response": {"result": msg.content or ""}}}],
                })
                continue
            role = "model" if msg.role == Role.ASSISTANT else "user"
            contents.append({"role": role, "parts": [{"text": msg.content or ""}]})
        return contents

    @staticmethod
    def _gemini_tools_formatter(tools: Optional[List[Dict[str, Any]]]) -> Optional[list]:
        if not tools:
            return None
        function_declarations = []
        for tool in tools or []:
            fn = tool.get("function", {})
            function_declarations.append({
                "name": fn.get("name"),
                "description": fn.get("description"),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        return [types.Tool(function_declarations=function_declarations)]

    @staticmethod
    def _gemini_get_first_content(response: Any) -> Any:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            return getattr(candidates[0], "content", None)
        return None
