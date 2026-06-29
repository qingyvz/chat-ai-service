import hashlib
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

PlanStepStatus = Literal["pending", "in_progress", "completed", "failed"]
PlanStatus = Literal["awaiting_review", "executing", "completed"]


class PlanStep(BaseModel):
    """todolist 单项"""
    step_id: str
    title: str
    description: str = ""
    status: PlanStepStatus = "pending"
    result_summary: Optional[str] = None


class Plan(BaseModel):
    """PlanMode 计划文件 DTO：真相源在 ai-asset（owner 维度、不绑会话），本服务持有内存态并热缓存到 Redis"""
    plan_id: str
    user_id: str
    resource_id: Optional[str] = None
    object_key: Optional[str] = None
    file_name: str = ""
    status: PlanStatus = "awaiting_review"
    content: str = ""
    steps: List[PlanStep] = Field(default_factory=list)
    content_hash: str = ""
    version: int = 1

    def steps_payload(self) -> List[dict]:
        return [{"id": s.step_id, "title": s.title, "status": s.status} for s in self.steps]

    def render_markdown(self) -> str:
        """渲染成可读 markdown（正文 + todolist），供 resource 内容与手改比对"""
        todo = "\n".join(f"- [{'x' if s.status == 'completed' else ' '}] {s.title}" for s in self.steps)
        return f"{self.content.rstrip()}\n\n## todolist\n{todo}\n" if todo else f"{self.content.rstrip()}\n"

    def compute_hash(self) -> str:
        """对渲染内容取 hash，用于检测用户手改"""
        return hashlib.sha256(self.render_markdown().encode("utf-8")).hexdigest()
