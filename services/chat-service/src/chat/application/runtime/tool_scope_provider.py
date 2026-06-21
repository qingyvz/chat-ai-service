from typing import Any, List, Optional, Set, Tuple

from chat.application.agents import AgentToolAndSkillPolicy
from chat.application.tools import ToolScope
from chat.application.tools.core import ToolRegistry
from chat.application.tools.skill_tools.utils.skill_matcher import SkillMatcher
from chat.domain.entities.skill import SkillMeta

# Skill 工具默认不暴露；仅在本轮存在可展示 Skill 时整体解禁
_SKILL_TOOL_NAMES = frozenset({"load_skill", "load_skill_asset"})
# Session 工具默认不暴露；仅在本轮存在不可见的上下文历史时解禁（有 summary）
_SESSION_TOOL_NAMES = frozenset({"get_historical_chat_messages"})


class ToolScopeProvider:
    """共享服务：skill 匹配 + 派生 tool_scope（能力解析，收窄靠 subagent 的 spec policy 驱动）"""

    def __init__(self, skill_matcher: SkillMatcher, tool_registry: ToolRegistry) -> None:
        self._skill_matcher = skill_matcher
        self._tool_registry = tool_registry

    async def resolve(
        self,
        session_id: str,
        user_id: str,
        user_query: str,
        tool_and_skill_policy: AgentToolAndSkillPolicy,
        session_summary: Optional[str],
        user_defined_allow_tool_names: Optional[Set[str]] = None,
        user_defined_deny_tool_names: Optional[Set[str]] = None,
        user_defined_on_demand_skill_ids: Optional[Set[str]] = None,
        runtime_context: Optional[dict] = None,
    ) -> Tuple[ToolScope, List[SkillMeta]]:
        # 构建工具上下文（runtime_context 透传 subagent_spawner/父模型/父 spec 等给工具执行时读取）
        tool_context: dict[str, Any] = {
            "session_id": session_id,
            "user_id": user_id,
            **(runtime_context or {}),
        }

        # 构建 Skill 视图：返回本轮可展示给 LLM 的 Skill metadata，由 LLM 判断是否加载
        available_skills: List[SkillMeta] = []
        if tool_and_skill_policy.enable_use_tool and tool_and_skill_policy.enable_use_skill:
            # 若用户指定了 on_demand_skill_ids，则覆盖 agent 预设
            on_demand_skill_ids = user_defined_on_demand_skill_ids or tool_and_skill_policy.on_demand_skill_ids or set()
            available_skills = await self._skill_matcher.match(
                on_demand_skill_ids=on_demand_skill_ids,
                user_query=user_query,
                skill_match_top_k=tool_and_skill_policy.skill_match_top_k,
            )

        expose_tool_name_set = None
        if available_skills:
            expose_tool_name_set = set()
            expose_tool_name_set.update(_SKILL_TOOL_NAMES)
            # allowed_skill_ids 表示本轮展示给 LLM 的 Skill 白名单，工具执行前仍会校验
            tool_context["allowed_skill_ids"] = [s.skill_id for s in available_skills]

        if session_summary is not None:
            expose_tool_name_set.update(_SESSION_TOOL_NAMES)

        # 构建工具视图：expose_tool_name_set 仅在有可展示 Skill 时解禁 Skill 工具
        if not tool_and_skill_policy.enable_use_tool:
            # 若不启用 Tool，则 allow_tool_name_set 为空
            allow_tool_name_set: Set[str] = set()
        else:
            # 若用户指定了 allow_tool_names，则覆盖 agent 预设
            allow_tool_name_set = user_defined_allow_tool_names or tool_and_skill_policy.allow_tool_names or None

        # 若用户指定了 deny_tool_names，则覆盖 agent 预设
        deny_tool_name_set = user_defined_deny_tool_names or tool_and_skill_policy.deny_tool_names or None

        tool_scope = self._tool_registry.derive(
            tool_context=tool_context,
            expose_tool_name_set=expose_tool_name_set,
            allow_tool_name_set=allow_tool_name_set,
            deny_tool_name_set=deny_tool_name_set,
        )

        return tool_scope, available_skills
