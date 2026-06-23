from typing import List, Literal, Optional
from pydantic import BaseModel, Field

PlanStepStatus = Literal["pending", "in_progress", "completed", "failed"]


class PlanStep(BaseModel):
    step_id: str
    title: str
    description: str = ""
    depends_on: List[str] = Field(default_factory=list)  # 预留 DAG，v1 全空=线性
    status: PlanStepStatus = "pending"
    result_summary: Optional[str] = None


class Plan(BaseModel):
    plan_id: str
    steps: List[PlanStep] = Field(default_factory=list)
