from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone

# -----------------------------
# Input passed to agents
# -----------------------------
class AgentInput(BaseModel):
  step: str
  agent_name: str
  instruction: str
  inputs: List[str]
# -----------------------------
# Output returned by agents
# -----------------------------
class AgentOutput(BaseModel):
  step: int
  agent: str
  output_key: str
  content: str
# -----------------------------
# Agent Schemas
# -----------------------------
 