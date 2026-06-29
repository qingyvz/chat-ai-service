from typing import Protocol

from chat.application.agents import (
    AgentInfo,
    AgentMemoryPolicy,
    AgentResolver,
    AgentSpec,
    AgentToolAndSkillPolicy,
    DefaultAgentResolver,
)
from chat.domain.repositories import SessionRepository

# subagent 自身禁用此工具 → 结构上不能再派 subagent
SUBAGENT_TOOL_NAMES = frozenset({"call_subagent"})

_SUBAGENT_SYSTEM_PROMPT = (
    "You are an execution sub-agent dispatched by an orchestrator. Complete ONLY the single task given, "
    "using any prior results in context. Be concise and produce a direct, usable result. "
    "Do not plan further; do not delegate."
)


class AgentProvider(Protocol):
    """取本轮 Agent；环境可换：Session 走 Mongo/default，SubAgent 走 Redis"""
    async def resolve(self, session_id: str, user_id: str) -> AgentInfo | None:
        ...


class SessionAgentProvider:
    """会话场景：按 session 绑定的 agent_id 解析持久化 Agent"""

    def __init__(self, agent_resolver: AgentResolver | None, session_repo: SessionRepository) -> None:
        self._agent_resolver = agent_resolver or DefaultAgentResolver()
        self._session_repo = session_repo

    async def resolve(self, session_id: str, user_id: str) -> AgentInfo | None:
        session = await self._session_repo.get_session_for_user(session_id, user_id)
        return await self._agent_resolver.resolve(session.agent_id)


class StaticAgentProvider:
    """子任务场景：直接返回 call_subagent 当场构造的 subagent spec，不落库"""

    def __init__(self, agent_info: AgentInfo) -> None:
        self._agent_info = agent_info

    async def resolve(self, session_id: str, user_id: str) -> AgentInfo | None:
        return self._agent_info


def build_subagent_info(parent_spec: AgentSpec, role: str, subagent_id: str) -> AgentInfo:
    """subagent 本质就是个 Agent：执行向 system prompt + 收窄 tool/skill policy（禁 subagent 工具与 skill）+ 继承迭代上限"""
    deny = set(parent_spec.tool_and_skill_policy.deny_tool_names or set()) | set(SUBAGENT_TOOL_NAMES)
    narrowed = AgentToolAndSkillPolicy(
        enable_use_tool=parent_spec.tool_and_skill_policy.enable_use_tool,
        allow_tool_names=parent_spec.tool_and_skill_policy.allow_tool_names,
        deny_tool_names=deny,
        enable_use_skill=False,
    )
    spec = AgentSpec(
        system_prompt=_SUBAGENT_SYSTEM_PROMPT,
        auto_generate_title=False,
        billing_group_id=parent_spec.billing_group_id,
        agent_max_iterations=parent_spec.agent_max_iterations,
        model_policy=parent_spec.model_policy,
        tool_and_skill_policy=narrowed,
        memory_policy=AgentMemoryPolicy(
            enable_chat_memory=False,
            enable_long_term_memory=False,
            enable_chat_memory_summary=False,
        ),
    )
    return AgentInfo(
        agent_id=subagent_id,
        name=role,
        description="plan-execute executor subagent",
        version=0,
        spec=spec,
    )
