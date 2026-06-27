import hashlib
import json
from datetime import datetime, timezone
from typing import List, Literal, Optional

from beanie import Document
from pydantic import BaseModel, Field
from pymongo import ASCENDING, DESCENDING, IndexModel

PlanStepStatus = Literal["pending", "in_progress", "completed", "failed"]
PlanStatus = Literal["awaiting_review", "executing", "completed"]


class PlanStep(BaseModel):
    """todolist 单项"""
    step_id: str
    title: str
    description: str = ""
    status: PlanStepStatus = "pending"
    result_summary: Optional[str] = None


class Plan(Document):
    """PlanMode 计划文件：chat-service 侧真相源（内容 + 状态 + todolist），并注册为 resource 暴露给用户"""
    plan_id: str
    session_id: str
    user_id: str
    resource_id: Optional[str] = None
    file_name: str = ""
    status: PlanStatus = "awaiting_review"
    content: str = ""
    steps: List[PlanStep] = Field(default_factory=list)
    content_hash: str = ""
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "plan"
        indexes = [
            IndexModel([("session_id", ASCENDING), ("status", ASCENDING)]),
            IndexModel([("user_id", ASCENDING), ("updated_at", DESCENDING)]),
        ]

    def steps_payload(self) -> List[dict]:
        return [{"id": s.step_id, "title": s.title, "status": s.status} for s in self.steps]

    def render_markdown(self) -> str:
        """渲染成可读 markdown（正文 + todolist），供 resource 内容与手改比对"""
        todo = "\n".join(f"- [{'x' if s.status == 'completed' else ' '}] {s.title}" for s in self.steps)
        return f"{self.content.rstrip()}\n\n## todolist\n{todo}\n" if todo else f"{self.content.rstrip()}\n"

    def compute_hash(self) -> str:
        """对渲染内容取 hash，用于检测用户手改"""
        return hashlib.sha256(self.render_markdown().encode("utf-8")).hexdigest()
