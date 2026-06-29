from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from chat.application.agents import AgentInfo
from chat.application.chat_context_assembler import ChatContextAssembler
from chat.application.events import StreamEvent
from chat.application.tools import ToolScope
from chat.domain.entities import ChatMessage
from chat.domain.entities.skill import SkillMeta
from chat.domain.repositories.model_repo import ModelRequestInfo


@dataclass
class RawMaterials:
    """编排入口所需的上下文原料（由 ContextProvider/ToolScopeProvider 备好，组装时机交给策略）"""
    user_query: str
    system_prompt: str
    history_messages: List[ChatMessage]
    relevant_facts: List[str]
    session_summary: Optional[str]
    frontend_states: Optional[List[Dict[str, Any]]]
    available_skills: List[SkillMeta]


@dataclass
class OrchestrationContext:
    """一轮编排的共享上下文：runtime 备料、策略读料并回写记账（record_messages / usage_tokens）"""
    session_id: str
    user_id: str
    agent_info: AgentInfo
    model: ModelRequestInfo
    raw_materials: RawMaterials
    assembler: ChatContextAssembler
    tool_scope: ToolScope
    record_messages: List[ChatMessage] = field(default_factory=list)
    usage_tokens: int = 0
    plan_review_decision: Optional[str] = None   # PlanMode 审查决策：execute / change


class OrchestrationStrategy(ABC):
    """编排策略抽象：由 think_type 选具体实现（ReAct / PlanMode），产出领域 StreamEvent"""

    @abstractmethod
    def run(self, ctx: OrchestrationContext) -> AsyncIterator[StreamEvent]:
        ...
