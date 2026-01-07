from __future__ import annotations
from typing import Dict, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
from enum import Enum
from .data_models import OrchestratorPlan

class TaskStatus(str, Enum):
  """Allowed lifecycle states for a task."""
  # INVARIANT:
  # TaskState must never store raw agent outputs or large text.
  # Only summaries and execution metadata are allowed.

  PENDING = "pending"
  RUNNING = "running"
  COMPLETED = "completed"
  FAILED = "failed"

class TaskRuntimeState(BaseModel):
  """
  Runtime state for a single task.

  NOTE:
  - output is always a STRING
  - errors are simple strings
  - status enum enables monitoring & retries
  """

  model_config = ConfigDict(extra="forbid")

  step: int
  agent: str
  instruction: str

  status: TaskStatus = TaskStatus.PENDING
  output: Optional[str] = None
  summary: Optional[str] = None
  error: Optional[str] = None

class TaskState(BaseModel):
  """
  Global workflow state shared across execution.
  """

  model_config = ConfigDict(extra="forbid")

  user_request: str = ""
  plan: Optional[OrchestratorPlan] = None
  tasks: Dict[int, TaskRuntimeState] = Field(default_factory=dict)

  created_at: datetime = Field(default_factory=lambda:datetime.now(timezone.utc))
  updated_at: datetime = Field(default_factory=lambda:datetime.now(timezone.utc))

  # -------------------------
  # State mutation helpers
  # -------------------------

  def init_from_plan(self, plan: OrchestratorPlan, user_request):
    """Initialize runtime tasks from an Orchestrator plan."""
    self.plan = plan
    for task in plan.tasks:
      self.tasks[task.step] = TaskRuntimeState(
        step=task.step,
        agent=task.agent,
        instruction=task.instruction
      )
    self.user_request = user_request
    self.updated_at = datetime.now(timezone.utc)

  def mark_running(self, step: int):
    self.tasks[step].status = TaskStatus.RUNNING
    self.updated_at = datetime.now(timezone.utc)

  def mark_completed(self, step: int, summary: str, output: str):
    task = self.tasks[step]
    task.status = TaskStatus.COMPLETED
    task.output = output
    task.summary = summary
    self.updated_at = datetime.now(timezone.utc)

  def mark_failed(self, step: int, error: str):
    task = self.tasks[step]
    task.status = TaskStatus.FAILED
    task.error = error
    self.updated_at = datetime.now(timezone.utc)