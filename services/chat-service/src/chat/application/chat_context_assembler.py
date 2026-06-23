from typing import Any, Dict, List, Optional

from chat.domain.entities import ChatMessage, Role
from chat.domain.entities.skill import SkillMeta


class ChatContextAssembler:
    """纯组装工具箱（无状态、无 I/O）：把原料拼成发往 LLM 的消息列表，由策略调用"""

    def assemble_prompt(
        self,
        session_id: str,
        user_query: str,
        system_prompt: str,
        session_summary: Optional[str],
        history_messages: List[ChatMessage],
        relevant_facts: List[str],
        frontend_states: Optional[List[Dict[str, Any]]] = None,
        available_skills: Optional[List[SkillMeta]] = None,
    ) -> List[ChatMessage]:
        """组装最终发往 LLM 的消息列表"""

        # Message 列表初始化并加入 System Prompt
        messages: List[ChatMessage] = [
            ChatMessage(session_id=session_id, role=Role.SYSTEM, content=system_prompt)
        ]

        # 如果有摘要，将其注入为 user 消息，位于明细上下文之前
        if session_summary:
            messages.append(ChatMessage(
                session_id=session_id,
                role=Role.USER,
                content=f"[Conversation Summary so far]:\n{session_summary}",
            ))

        # 追加近期对话明细
        messages.extend(history_messages)

        # -- 以上消息在多轮对话中保持公共前缀，可命中缓存 --

        # 上下文块组装
        context_blocks: List[str] = []

        # 从 Mem0 召回的相关事实
        if relevant_facts:
            facts_text = "\n".join([f"- {fact}" for fact in relevant_facts])
            context_blocks.append(
                f"<relevant_user_memories>\n{facts_text}\n</relevant_user_memories>"
            )

        # Skill 提示：披露轻量 metadata，由 LLM 判断是否加载完整 SKILL.md
        if available_skills:
            context_blocks.append(
                f"<available_skills>\n{self._skills_block(available_skills)}\n</available_skills>"
            )

        # 前端上下文注入：筛选出未禁用且有 value 的元素
        active_frontend_states = [state for state in (frontend_states or []) if not state.get("disabled", False) and state.get("value")]
        if active_frontend_states:
            context_lines = [f'<context key="{state["key"]}">\n{state["value"]}\n</context>' for state in active_frontend_states]
            context_blocks.append(
                "<user_frontend_context>\n" + "\n".join(context_lines) + "\n</user_frontend_context>"
            )

        # 用户最新输入的问题
        if context_blocks:
            final_user_content = (
                    "[Application-provided context]\n"
                    "The following context is provided by the application. "
                    "Use it as background information, but the user's actual request is in <user_query>.\n\n"
                    + "\n\n".join(context_blocks)
                    + f"\n\n<user_query>\n{user_query}\n</user_query>"
            )
        else:
            final_user_content = user_query

        # 该 Message 不持久化
        messages.append(ChatMessage(
            session_id=session_id,
            role=Role.USER,
            content=final_user_content,
        ))

        return messages

    @staticmethod
    def _skills_block(available_skills: List[SkillMeta]) -> str:
        skill_lines = [
            f'- id="{skill.skill_id}" name="{skill.name}" : {skill.description}'
            for skill in available_skills
        ]
        return (
                "The following skills are available in this turn as lightweight metadata. "
                "Each skill contains detailed domain instructions in SKILL.md and may include supporting assets.\n"
                "Strict rules:\n"
                "1. If the user explicitly asks to use one of the listed skills by id or name, call `load_skill` for that skill.\n"
                "2. Otherwise, call `load_skill` only when a listed skill is directly useful for the current request. Do not load speculatively.\n"
                "3. To load a skill, call `load_skill` with `skill_id` exactly as listed below.\n"
                "4. After loading, the returned SKILL.md is mandatory for the current task. Follow its Scope, Output Format, and Constraints precisely.\n"
                "5. Call `load_skill_asset` only after loading a skill, and only if the loaded SKILL.md explicitly requires a listed asset.\n"
                "6. If none of the skills apply, ignore this list and answer normally.\n\n"
                "Skills:\n"
                + "\n".join(skill_lines)
        )
