from __future__ import annotations
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, field_validator

class AgentName(str, Enum):
  Orchestrator = "Orchestrator"
  Professor = "Professor"
  Researcher = "Researcher"
  Communicator = "Communicator"
  Secretary = "Secretary"
  Accountant = "Accountant"
  Responder = "Responder"
  Synthesizer = "Synthesizer"

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
  inputs: List[str] = Field(default_factory=list)
  output: str = Field(..., min_length=1)
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

class MessageRole(str, Enum):
  USER = "user"
  ASSISTANT = "assistant"
  SYSTEM = "system"

class MemoryNamespace(str, Enum):
  CONVERSATION = "conversation"  # Conversation history
  USER_PREFERENCES = "preferences"  # User preferences
  DOCUMENT_CONTEXT = "documents"  # Document references
  AGENT_OUTPUTS = "outputs"  # Agent outputs for reuse

class Message(BaseModel):
  """Single message in conversation history."""
  
  model_config = ConfigDict(extra="forbid")

  role: MessageRole
  agent: Optional[AgentName] = None
  content: str = Field(..., min_length=1)
  metadata: Dict[str, Any] = Field(default_factory=dict)
  timestamp: datetime = Field(default_factory=lambda:datetime.now(timezone.utc))

  @field_validator("content")
  def content_not_empty(cls, content: str) -> str:
    if not content.strip():
      raise ValueError("Content cannot be empty")
    return content