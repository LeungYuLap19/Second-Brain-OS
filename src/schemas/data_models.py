from __future__ import annotations
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator

class AgentName(str, Enum):
  Orchestrator = "Orchestrator"
  Professor = "Professor"
  Researcher = "Researcher"
  Communicator = "Communicator"
  Secretary = "Secretary"
  Accountant = "Accountant"
  Responder = "Responder"
  Distiller = "Distiller"

class TaskSpec(BaseModel):
  """
  Minimal immutable task definition produced by the Orchestrator.

  OUTPUT CONTRACT:
  - All agent outputs are expected to be STRINGS
  - No hidden or dynamic fields allowed
  """

  model_config = ConfigDict(extra="forbid")
  
  step: int = Field(..., ge=1)
  agent: AgentName
  instruction: str = Field(..., min_length=1)
  can_run_in_parallel: bool = False

class OrchestratorPlan(BaseModel):
  """
  Container for all TaskSpec items returned by the Orchestrator.
  """

  model_config = ConfigDict(extra="forbid")

  tasks: List[TaskSpec]

  @field_validator("tasks")
  def validate_steps_are_sequential(cls, tasks: List[TaskSpec]):
    steps = [t.step for t in tasks]
    if steps != sorted(steps):
      raise ValueError("Task steps must be in ascending order")
    return tasks