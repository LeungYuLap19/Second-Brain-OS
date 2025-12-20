import json
from typing import Dict, Any, Optional
from .base_agent import BaseAgent


class OrchestratorAgent(BaseAgent):
  """
  The central planning brain.

  Produces STRICT JSON ONLY:

  {
    "tasks": [
      {
        "step": 1,
        "agent": "<Exact Agent Name>",
        "instruction": "<string>",
        "inputs": [],
        "output": "step1.<Exact Agent Name>",
        "can_run_in_parallel": false
      }
    ]
  }
  """

  def __init__(self):
    # Init BaseAgent
    # Load config and prompt for Orchestrator
    super().__init__("Orchestrator")

  def run(self, user_input: str) -> Dict[str, Any]:
    """
    Calls the underlying LLM and returns a parsed task plan.
    Automatically retries once if JSON is malformed.
    """
    raw_output = super().run(user_input)

    parsed = self._safe_parse_json(raw_output)
    if parsed is not None:
      return parsed

    # -----------------------------
    # Auto-repair attempt
    # -----------------------------
    repair_prompt = (
      "The previous output was NOT valid JSON.\n\n"
      "Return ONLY valid JSON that strictly matches this schema:\n\n"
      "{\n"
      '  "tasks": [\n'
      "    {\n"
      '      "step": 1,\n'
      '      "agent": "",\n'
      '      "instruction": "",\n'
      '      "inputs": [],\n'
      '      "output": "",\n'
      '      "can_run_in_parallel": false\n'
      "    }\n"
      "  ]\n"
      "}\n\n"
      "Rules:\n"
      "- Output JSON only\n"
      "- No comments\n"
      "- No markdown\n\n"
      f"Invalid output:\n{raw_output}\n"
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
  # def _prepare_input(self, user_input: str) -> str:
  #   """
  #   Constructs the prompt for the Orchestrator LLM.
  #   """
  #   return (
  #     "User request:\n"
  #     f"{user_input}\n\n"
  #     "Produce a deterministic task plan in STRICT JSON only."
  #   )

  # -------------------------------------
  # Helper: Safe JSON parse
  # -------------------------------------
  def _safe_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to parse JSON.
    Returns dict if valid, otherwise None.
    """
    try:
      return json.loads(text)
    except json.JSONDecodeError:
      return None