from enum import StrEnum

from pydantic import BaseModel, Field
from uuid import uuid4
from datetime import datetime
from typing import Optional, Literal

class TaskStatus(StrEnum):
    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"


class TaskPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class TaskLang(StrEnum):
    PL = "pl"
    EN = "en"

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the task")
    description: str = Field(..., description="Description of the task", min_length=1)
    relation: Optional[str] = Field(None, description="Relation between tasks")
    status: TaskStatus = Field(TaskStatus.TODO, description="Status of the task")
    priority: TaskPriority = Field(TaskPriority.NORMAL, description="Priority of the task")
    created_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"), description="Creation timestamp of the task")
    updated_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"), description="Update timestamp of the task")
    lang: TaskLang = Field(TaskLang.PL, description="Language of the task")
