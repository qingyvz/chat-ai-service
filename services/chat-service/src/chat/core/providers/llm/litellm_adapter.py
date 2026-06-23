import uuid
from typing import Any, AsyncGenerator, AsyncIterable, Dict, List, Optional, cast

import litellm

from chat.core.config.app_settings import settings
from chat.core.config.bootstrap_settings import bootstrap_settings
from chat.domain.entities import ChatMessage, Role
from chat.domain.entities.message import ToolCallMessage
from chat.domain.entities.provider import ProviderType
from chat.domain.error_codes import ChatErrorCode
from chat.domain.interfaces import LLMProvider
from chat.domain.interfaces.llm import (
    LLMCompletionResult,
    LLMEventType,
    LLMStreamEvent,
    LLMUsage,
    TextCompletionProvider,
)
from chat.domain.repositories.model_repo import ModelRequestInfo
from common.core.exceptions import ServiceException

from .utils import json_object, read_provider_value

litellm.telemetry = False

_is_debug = bootstrap_settings.LOG_LEVEL.upper() == "DEBUG"
litellm.set_verbose = _is_debug
litellm.suppress_debug_info = not _is_debug


class LiteLLMAdapter(LLMProvider, TextCompletionProvider):
    """LiteLLM 进程内 OpenAI-compatible 调用；归一化产出 LLMStreamEvent，并实现非流式 TextCompletionProvider"""

    def __init__(self):
        self._default_api_base = settings.LLM_BASE_URL
        self._default_api_key = settings.LLM_API_KEY

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LITELLM_OPENAI_COMPATIBLE

    def runtime_options_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "json_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                    "top_p": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "presence_penalty": {"type": "number", "minimum": -2, "maximum": 2},
                    "frequency_penalty": {"type": "number", "minimum": -2, "maximum": 2},
                    "seed": {"type": "integer"},
                },
            },
            "defaults": {"temperature": 0.7},
        }

    @staticmethod
    def _litellm_messages_formatter(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        # assistant 优先回放 LiteLLM 自己保存的原生载荷，其余 provider/消息降级为可见文本
        formatted_messages = []
        for message in messages:
            if (message.role == Role.ASSISTANT and message.model_info
                    and message.model_info.provider_type == ProviderType.LITELLM_OPENAI_COMPATIBLE
                    and message.provider_payload):
                formatted_messages.append(message.provider_payload["message"])
                continue
            if message.role == Role.TOOL:
                formatted_messages.append({
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "name": message.name,
                    "content": message.content or "",
                })
                continue
            formatted_messages.append({"role": message.role.value, "content": message.content or ""})
        return formatted_messages

    @staticmethod
    def _to_openai_compatible_model(model_name: str) -> str:
        if "/" in model_name:
            return model_name
        return f"openai/{model_name}"

    async def chat_completion(
            self,
            messages: List[ChatMessage],
            model_name: str,
            temperature: float = 0.7,
            tools: Optional[List[Dict[str, Any]]] = None,
            api_base: Optional[str] = None,
            api_key: Optional[str] = None,
    ) -> LLMCompletionResult:
        formatted_messages = self._litellm_messages_formatter(messages)
        litellm_model = self._to_openai_compatible_model(model_name)
        try:
            response = await litellm.acompletion(
                model=litellm_model,
                messages=formatted_messages,
                stream=False,
                temperature=temperature,
                tools=tools,
                drop_params=True,
                api_base=api_base or self._default_api_base,
                api_key=api_key or self._default_api_key,
            )
            usage = getattr(response, "usage", None)
            usage_tokens = getattr(usage, "total_tokens", 0) if usage else 0
            return LLMCompletionResult(raw=response, usage_tokens=int(usage_tokens))
        except litellm.ContextWindowExceededError:
            raise ServiceException(ChatErrorCode.CONTEXT_LIMIT_EXCEEDED)
        except Exception as e:
            raise ServiceException(ChatErrorCode.LLM_GENERATION_FAILED, custom_msg=f"Provider Error: {e}")

    async def stream_chat_completion(
            self,
            messages: List[ChatMessage],
            model_request: ModelRequestInfo,
            tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        formatted_msgs = self._litellm_messages_formatter(messages)
        litellm_model = self._to_openai_compatible_model(model_request.model_name)

        token_usage = 0
        tool_acc: dict[int, dict[str, str]] = {}
        assistant_text = ""
        try:
            response = await litellm.acompletion(
                model=litellm_model,
                messages=formatted_msgs,
                stream=True,
                stream_options={"include_usage": True},
                tools=tools,
                drop_params=True,
                api_base=model_request.base_url or self._default_api_base,
                api_key=model_request.api_key or self._default_api_key,
                **(model_request.runtime_options or {}),
            )
            stream = cast(AsyncIterable[Any], response)

            async for chunk in stream:
                usage = read_provider_value(chunk, "usage", {}) or {}
                token_usage = int(read_provider_value(usage, "total_tokens", token_usage) or token_usage)

                choices = read_provider_value(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = read_provider_value(choices[0], "delta", {}) or {}
                # 思考增量
                reasoning = read_provider_value(delta, "reasoning_content")
                if reasoning:
                    yield LLMStreamEvent(type=LLMEventType.REASONING_DELTA, delta=reasoning)
                # 文本增量
                if getattr(delta, "content", None):
                    assistant_text += delta.content
                    yield LLMStreamEvent(type=LLMEventType.TEXT_DELTA, delta=delta.content)
                # 工具调用分片累积
                if getattr(delta, "tool_calls", None):
                    for tool_call_delta in delta.tool_calls:
                        idx = tool_call_delta.index
                        acc = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tool_call_delta.id:
                            acc["id"] = tool_call_delta.id
                        if tool_call_delta.function:
                            if tool_call_delta.function.name:
                                acc["name"] += tool_call_delta.function.name
                            if tool_call_delta.function.arguments:
                                acc["arguments"] += tool_call_delta.function.arguments
        except litellm.ContextWindowExceededError:
            raise ServiceException(ChatErrorCode.CONTEXT_LIMIT_EXCEEDED)
        except Exception as e:
            raise ServiceException(ChatErrorCode.LLM_GENERATION_FAILED, custom_msg=f"Provider Error: {e}")

        # 计费
        if token_usage:
            yield LLMStreamEvent(type=LLMEventType.USAGE, usage=LLMUsage(output_tokens=int(token_usage)))

        # 解析工具调用
        tool_calls: list[ToolCallMessage] = []
        tool_call_payloads = []
        for idx in sorted(tool_acc.keys()):
            acc = tool_acc[idx]
            tool_call_payloads.append({
                "id": acc["id"],
                "type": "function",
                "function": {"name": acc["name"], "arguments": acc["arguments"]},
            })
            tool_calls.append(ToolCallMessage(
                call_id=acc["id"] or f"call_{uuid.uuid4().hex}",
                name=acc["name"],
                arguments=json_object(acc["arguments"]),
            ))
        if tool_calls:
            yield LLMStreamEvent(type=LLMEventType.TOOL_CALLS, tool_calls=tool_calls)

        # 保存 OpenAI-compatible assistant message，供下一轮协议回放
        assistant_message: Dict[str, Any] = {"role": "assistant", "content": assistant_text or None}
        if tool_call_payloads:
            assistant_message["tool_calls"] = tool_call_payloads
        yield LLMStreamEvent(type=LLMEventType.STATE, provider_payload={"message": assistant_message})
