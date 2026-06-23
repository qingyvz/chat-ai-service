from enum import Enum
import jieba
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime, timezone
from beanie import Document, PydanticObjectId
from pydantic import Field, ConfigDict, BaseModel
from pymongo import IndexModel, ASCENDING

from chat.domain.entities.model import ModelFamily, ModelScope
from chat.domain.entities.provider import ProviderType

if TYPE_CHECKING:
    from chat.domain.repositories.model_repo import ModelRequestInfo


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCallMessage(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MessageModelInfo(BaseModel):
    """消息持久化用的模型安全快照（assistant 消息回放/计费用）"""
    model_id: str
    provider_id: str
    provider_type: ProviderType
    model_family: ModelFamily
    model_name: str
    scope: ModelScope
    support_tools: bool
    context_window_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    runtime_options: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_model_request(cls, model_request: "ModelRequestInfo") -> "MessageModelInfo":
        return cls(
            model_id=str(model_request.model_id),
            provider_id=str(model_request.provider_id),
            provider_type=model_request.provider_type,
            model_family=model_request.model.model_family,
            model_name=model_request.model_name,
            scope=model_request.scope,
            support_tools=model_request.support_tools,
            context_window_tokens=model_request.context_window_tokens,
            max_output_tokens=model_request.max_output_tokens,
            runtime_options=model_request.runtime_options or {},
        )


class ChatMessage(Document):
    """单条消息实体（Beanie Document，映射到 chat_messages 集合）"""
    session_id: str
    role: Role
    model_id: Optional[PydanticObjectId] = None  # 生成该消息所用的模型 _id，仅 assistant 消息必填
    # provider 归一化新增（assistant 消息）：模型安全快照 + 原生载荷回放 + 用量
    model_info: Optional[MessageModelInfo] = None
    provider_payload: Optional[Dict[str, Any]] = None  # LLMProvider 原生载荷，用于历史回放
    token_usage: int = 0
    content: Optional[str] = None   # 大模型在返回 tool_calls 时 content 经常为 None
    reasoning_content: Optional[str] = None  # 大模型的推理/思考内容（DeepSeek R1 等）
    search_tokens: Optional[str] = None # 专门用于规避 MongoDB 中文分词缺陷的隐藏字段

    token_count: Optional[int] = None # 消息内容对应的 Token 数，随消息创建时一次性计算并持久化
    metadata: Dict[str, Any] = Field(default_factory=dict)

    tool_calls: Optional[List[ToolCallMessage]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    # 仅本轮工作内可见标识
    persisted_output_placeholder: str | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=False)

    class Settings:
        name = "chat_message"  # MongoDB 集合名
        indexes = [
            # 按会话拉取历史记录的核心查询路径，防全表扫描
            IndexModel([("session_id", ASCENDING), ("created_at", ASCENDING)]),
            # 支持 Tool Calling 的全文关键词检索
            IndexModel([("search_tokens", "text")]),
        ]

    @property
    def is_human(self) -> bool:
        return self.role == Role.USER


    def build_search_tokens(self) -> None:
        """
        在保存前调用此方法。
        使用搜索引擎模式的分词（cut_for_search），最大化召回率。
        例如："软件工程架构" -> "软件 工程 软件工程 架构"
        """
        if self.content:
            # 过滤掉单字和标点符号，用空格拼接
            words = jieba.cut_for_search(self.content)
            self.search_tokens = " ".join([w for w in words if len(w.strip()) > 1])
