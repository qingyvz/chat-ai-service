import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    import dashscope
except ImportError:  # SDK 未装：adapter 仍可注册，调用时再报错（当前 qwen 走 litellm 路径）
    dashscope = None

from chat.domain.entities import ChatMessage, Role
from chat.domain.entities.provider import ProviderType
from chat.domain.error_codes import ChatErrorCode
from chat.domain.interfaces import LLMProvider
from chat.domain.interfaces.llm import LLMEventType, LLMStreamEvent, LLMUsage
from chat.domain.entities.message import ToolCallMessage
from chat.domain.repositories.model_repo import ModelRequestInfo
from common.core.exceptions import ServiceException

from .utils import json_object, read_provider_value, without_none


class QwenAdapter(LLMProvider):
    """Qwen 官方阿里云百炼 DashScope Python SDK 适配器"""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ALIBABA

    def runtime_options_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "json_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "temperature": {"type": "number", "minimum": 0, "exclusiveMaximum": 2},
                    "top_p": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "top_k": {"type": "integer", "minimum": 0},
                    "enable_thinking": {"type": "boolean"},
                    "thinking_budget": {"type": "integer", "minimum": 0},
                    "presence_penalty": {"type": "number", "minimum": -2, "maximum": 2},
                    "repetition_penalty": {"type": "number", "minimum": 0},
                    "seed": {"type": "integer"},
                },
            },
            "defaults": {"temperature": 0.7, "enable_thinking": True},
        }

    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model_request: ModelRequestInfo,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        if dashscope is None:
            raise ServiceException(ChatErrorCode.LLM_GENERATION_FAILED, custom_msg="dashscope SDK 未安装")

        qwen_messages = self._qwen_messages_formatter(messages)
        request_kwargs: dict[str, Any] = {
            "api_key": model_request.api_key,
            "model": model_request.model_name,
            "messages": qwen_messages,
            "result_format": "message",
            "stream": True,
            "incremental_output": True,
            "tools": tools,  # Qwen function calling 使用 OpenAI-compatible tools schema
            **model_request.runtime_options,
        }

        assistant_text = ""
        reasoning_text = ""
        tool_calls: list[ToolCallMessage] = []
        tool_call_payloads = []
        token_usage = 0
        try:
            responses = dashscope.Generation.call(**without_none(request_kwargs))
            for response in responses:
                status_code = read_provider_value(response, "status_code")
                if status_code and int(status_code) >= 400:
                    message = read_provider_value(response, "message", "DashScope request failed")
                    raise ServiceException(ChatErrorCode.LLM_GENERATION_FAILED, custom_msg=f"Qwen Provider Error: {message}")

                output = read_provider_value(response, "output", {}) or {}
                usage = read_provider_value(response, "usage", {}) or {}
                token_usage = int(read_provider_value(usage, "total_tokens", token_usage) or token_usage)

                choices = read_provider_value(output, "choices", []) or []
                if not choices:
                    continue
                message = read_provider_value(choices[0], "message", {}) or {}
                reasoning = read_provider_value(message, "reasoning_content")
                if reasoning:
                    reasoning_text += reasoning
                    yield LLMStreamEvent(type=LLMEventType.REASONING_DELTA, delta=reasoning)
                content = read_provider_value(message, "content")
                if content:
                    assistant_text += content
                    yield LLMStreamEvent(type=LLMEventType.TEXT_DELTA, delta=content)

                # Qwen tool_calls 的 function.arguments 已经是完整 JSON 字符串
                for call in read_provider_value(message, "tool_calls", []) or []:
                    payload = call if isinstance(call, dict) else {}
                    function = payload.get("function", {})
                    tool_call_payloads.append(payload)
                    tool_calls.append(ToolCallMessage(
                        call_id=payload.get("id") or f"call_{uuid.uuid4().hex}",
                        name=function.get("name", ""),
                        arguments=json_object(function.get("arguments", "{}")),
                    ))
        except ServiceException:
            raise
        except Exception as e:
            raise ServiceException(ChatErrorCode.LLM_GENERATION_FAILED, custom_msg=f"Qwen Provider Error: {e}")

        if token_usage:
            yield LLMStreamEvent(type=LLMEventType.USAGE, usage=LLMUsage(output_tokens=token_usage))
        if tool_calls:
            yield LLMStreamEvent(type=LLMEventType.TOOL_CALLS, tool_calls=tool_calls)

        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": assistant_text or None,
            "reasoning_content": reasoning_text or None,
        }
        if tool_call_payloads:
            assistant_message["tool_calls"] = tool_call_payloads
        yield LLMStreamEvent(type=LLMEventType.STATE, provider_payload={"message": assistant_message})

    @staticmethod
    def _qwen_messages_formatter(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        result = []
        for msg in messages:
            if (msg.role == Role.ASSISTANT and msg.model_info
                    and msg.model_info.provider_type == ProviderType.ALIBABA and msg.provider_payload):
                result.append(msg.provider_payload["message"])
                continue
            if msg.role == Role.TOOL:
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id, "name": msg.name, "content": msg.content or "",
                })
                continue
            result.append({"role": msg.role.value, "content": msg.content or ""})
        return result
