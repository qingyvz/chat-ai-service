from chat.domain.entities import Plan

PLAN_FORMAT_DIRECTIVE = (
    "You are in PLAN MODE. Do NOT execute the task yet. First produce a PLAN file by calling create_file once.\n"
    "The plan `content` (markdown) MUST contain these sections in order:\n"
    "## 背景\n阐释你对本需求的理解。\n"
    "## 任务\n要达成的目标。\n"
    "## 注意事项\n执行约束与风险提示。若用户未提供，请基于本计划实际情况生成具体、克制的注意事项，"
    "防止幻觉或危险/不可逆操作（如删除数据、对外发布、越权访问）。\n"
    "## 思路\n达成任务的总体思路。\n"
    "Pass a `steps` array as the todolist (ordered, each with a short title). "
    "After create_file succeeds, briefly tell the user the plan is ready for review (execute / change)."
)

EXECUTE_DIRECTIVE = (
    "The user APPROVED the plan below. Execute it now with the available tools, following the plan.\n"
    "As you work each todolist step, call update_plan(step_id, status): set 'in_progress' when you start a step, "
    "'completed' when done (or 'failed' if it cannot be done). When all steps are done, write the final answer."
)


def execute_plan_block(plan: Plan) -> str:
    lines = [f"- {s.step_id}: {s.title}（{s.status}）" for s in plan.steps]
    return "已批准的计划正文：\n" + plan.content.rstrip() + "\n\ntodolist：\n" + "\n".join(lines)


def change_directive(feedback: str, manually_edited: bool) -> str:
    head = (
        "The user requested CHANGES to the plan. Revise it by calling update_plan "
        "(rewrite `content` and/or adjust steps). Stay in plan mode; do NOT execute.\n"
    )
    if manually_edited:
        head += "Note: the user also edited the plan content directly; the current content is shown below.\n"
    if feedback:
        head += f"User feedback: {feedback}\n"
    return head


def change_plan_block(plan: Plan) -> str:
    return "当前计划内容：\n" + plan.content.rstrip()
