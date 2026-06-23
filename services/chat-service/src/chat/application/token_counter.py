import json
from typing import Any, Dict, List, Optional

import litellm

from chat.domain.entities import ChatMessage, Role


class TokenCounter:
    """基于 LiteLLM 的 Token 估算器，provider usage 缺失时作为兜底来源（从 LLMProvider 拆出）"""

    async def count_text(self, text: str, model_name: str = "gpt-4o") -> int:
        try:
            return litellm.token_counter(model=self._to_openai_compatible_model(model_name), text=text)
        except Exception:
            return max(1, len(text))

    async def count_messages(
        self,
        messages: List[ChatMessage],
        model_name: str = "gpt-4o",
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        try:
            result = await litellm.acount_tokens(
                model=self._to_openai_compatible_model(model_name),
                messages=self._convert_messages(messages),
                tools=tools,
            )
            return int(getattr(result, "total_tokens", 0) or 0)
        except Exception:
            payload = self._convert_messages(messages)
            if tools:
                payload.append({"tools": tools})
            return max(1, len(json.dumps(payload, ensure_ascii=False, default=str)))

    @staticmethod
    def _to_openai_compatible_model(model_name: str) -> str:
        if "/" in model_name:
            return model_name
        return f"openai/{model_name}"

    @staticmethod
    def _convert_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        formatted_messages = []
        for message in messages:
            if message.role == Role.ASSISTANT:
                payload = {"role": message.role.value, "content": message.content, "reasoning": message.reasoning_content}
                if message.tool_calls:
                    payload["tool_calls"] = [
                        {
                            "id": tool_call.call_id,
                            "type": "function",
                            "function": {"name": tool_call.name, "arguments": tool_call.arguments},
                        }
                        for tool_call in message.tool_calls
                    ]
            else:
                payload = {"role": message.role.value, "content": message.content}
                if message.role == Role.TOOL:
                    if message.tool_call_id:
                        payload["tool_call_id"] = message.tool_call_id
                    if message.name:
                        payload["name"] = message.name
            formatted_messages.append(payload)
        return formatted_messages
