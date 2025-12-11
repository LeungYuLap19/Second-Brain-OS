import json
from typing import Dict, Any
from .base_agent import BaseAgent

class OrchestratorAgent(BaseAgent):
  """
  The central routing brain.
  Produces strict JSON:
  {
    "next_agent": "<agent_name_or_synthesizer>",
    "task_plan": [...],
    "current_step": "<string>",
    "notes_for_agent": "<string>",
    "is_final": false
  }
  """

  def __init__(self):
    super().__init__("Orchestrator")

  def run(self, user_input: str, task_state: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Calls the underlying LLM and returns a parsed JSON dict.
    Automatically retries once if JSON is malformed.
    """

    """
    Feed user inputs and task state in LLM
    Parse LLM JSON output
    """
    combined_input = self._prepare_input(user_input, task_state)
    raw_output = super().run(combined_input)
    parsed = self._safe_parse_json(raw_output)

    """Check if LLM returns JSON"""
    if parsed is not None:
      return parsed 
    
    """Auto-retry (ask model to fix JSON)"""
    repair_prompt = (
      "The previous output was not valid JSON. "
      "Please return ONLY valid JSON with the required fields:\n\n"
      "{\n"
      '  "next_agent": "",\n'
      '  "task_plan": [],\n'
      '  "current_step": "",\n'
      '  "notes_for_agent": "",\n'
      '  "is_final": false\n'
      "}\n\n"
      f"Here is the invalid output:\n{raw_output}\n"
    )

    repaired_raw = super().run(repair_prompt)
    repaired = self._safe_parse_json(repaired_raw)

    if repaired is None:
      raise ValueError(
        "Orchestrator failed to produce valid JSON after repair attempt.\n"
        f"Raw Output:\n{raw_output}"
      )

    return repaired
  
  # -------------------------------------
  # Helper: Build LLM input
  # -------------------------------------
  def _prepare_input(self, user_input: str, task_state: Dict[str, Any]) -> str:
    """
    Constructs a combined prompt including task state.
    """
    if task_state is None:
      return user_input

    # Convert task_state to readable text for the LLM
    ts_pretty = json.dumps(task_state, indent=2)

    return (
      "User request:\n"
      f"{user_input}\n\n"
      "Current Task State:\n"
      f"{ts_pretty}\n\n"
      "Provide the next JSON routing decision."
    )
  
  # -------------------------------------
  # Helper: Safe JSON parse
  # -------------------------------------
  def _safe_parse_json(self, text: str) -> Dict[str, Any]:
    """
    Returns dict or None if invalid.
    """
    try:
      return json.loads(text)
    except json.JSONDecodeError:
      return None