from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
from enum import Enum
from .data_models import OrchestratorPlan

class TaskStatus(str, Enum):
  """Allowed lifecycle states for a task."""

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
  inputs: List[str] = Field(default_factory=list)

  status: TaskStatus = TaskStatus.PENDING
  output: Optional[str] = None
  error: Optional[str] = None

class TaskState(BaseModel):
  """
  Global workflow state shared across execution.


  DESIGN GOALS:
  - Extremely simple
  - Easy to reason about
  - Safe for beginners
  """

  model_config = ConfigDict(extra="forbid")

  user_request: str = ""

  # Orchestrator output (validated once)
  plan: Optional[OrchestratorPlan] = None

  # Runtime task tracking
  tasks: Dict[int, TaskRuntimeState] = Field(default_factory=dict)

  # Collected agent outputs (ALL STRINGS)
  results: Dict[str, str] = Field(default_factory=dict)

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
        instruction=task.instruction,
        inputs=task.inputs,
      )
    self.user_request = user_request
    self.updated_at = datetime.now(timezone.utc)

  def mark_running(self, step: int):
    self.tasks[step].status = TaskStatus.RUNNING
    self.updated_at = datetime.now(timezone.utc)

  def mark_completed(self, step: int, output_key: str, output: str):
    task = self.tasks[step]
    task.status = TaskStatus.COMPLETED
    task.output = output
    self.results[output_key] = output
    self.updated_at = datetime.now(timezone.utc)

  def mark_failed(self, step: int, error: str):
    task = self.tasks[step]
    task.status = TaskStatus.FAILED
    task.error = error
    self.updated_at = datetime.now(timezone.utc)