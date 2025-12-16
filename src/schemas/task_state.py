from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone

# -----------------------------
# Agent execution status
# -----------------------------
class AgentStatus(Enum):
  IDLE = "idle"
  RUNNING = "running"
  COMPLETED = "completed"
  FAILED = "failed"

# -----------------------------
# Per-agent state
# -----------------------------
class AgentState(BaseModel):
  agent_name: str
  status: AgentStatus = AgentStatus.IDLE
  input: Optional[Dict[str, Any]] = None
  output: Optional[Any] = None
  error: Optional[str] = None
  started_at: Optional[datetime] = None
  finished_at: Optional[datetime] = None

# -----------------------------
# Global LangGraph State
# -----------------------------
class taskState(BaseModel):
  """
  Canonical LangGraph State.
  This object is passed between all nodes.
  """
  
  task_id: str
  user_request: str
  intent: Optional[str] = None
  routing_decision: Optional[str] = None
  agent_states: Dict[str, AgentState] = Field(default_factory=dict)
  # 
  final_output: Optional[str] = None
  is_complete: bool = False
  created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
  updated_at: datetime = Field(default_factory=datetime.now(timezone.utc))

  # -------------------------
  # Convenience helpers
  # -------------------------
  def mark_agent_running(self, agent_name: str, input_payload: Dict[str, Any]):
    self.agent_states[agent_name] = AgentState(
      agent_name=agent_name,
      status=AgentStatus.RUNNING,
      input=input_payload,
      started_at=datetime.now(timezone.utc)
    )

  def mark_agent_completed(self, agent_name: str, output: Any):
    agent_state = self.agent_states.get(agent_name)
    if agent_state:
      agent_state.status = AgentStatus.COMPLETED
      agent_state.output = output
      agent_state.finished_at = datetime.now(timezone.utc)

  def mark_agent_failed(self, agent_name: str, error: str):
    agent_state = self.agent_states.get(agent_name)
    if agent_state:
      agent_state.status = AgentStatus.FAILED
      agent_state.error = error
      agent_state.finished_at = datetime.now(timezone.utc)