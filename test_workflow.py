from src.agents.orchestrator import OrchestratorAgent
from src.schemas.task_state import TaskState
from src.schemas.data_models import OrchestratorPlan
from src.orchestration.workflow_manager import WorkflowManager
import json

# -------------------------
# Mock agent runner
# -------------------------
def mock_agent_runner(agent_name: str, input_text: str) -> str:
    """
    Deterministic fake agent execution.
    Prints input for debugging.
    """
    print(f"\n--- MOCK RUN: {agent_name} ---")
    print(input_text)
    print("-----------------------------")

    return f"[MOCK OUTPUT from {agent_name}]"

# integration testing
def test_workflow():
  orch = OrchestratorAgent()

  user_request = "Summarize the PDF and then turn it into point form notes."
  plan = orch.run(user_request)
  plan = OrchestratorPlan.model_validate(plan)

  print("\n=== ORCHESTRATOR PLAN ===")
  for task in plan.tasks:
    print(task)

  initial_state = TaskState()
  initial_state.init_from_plan(plan=plan, user_request=user_request)

  manager = WorkflowManager(agent_runner=mock_agent_runner)
  graph = manager.build_graph(plan)
  app = graph.compile()
  final_state = app.invoke(initial_state)

  for step, task in final_state["tasks"].items():
    assert task.status.name == "COMPLETED"

  assert "final.Synthesizer" in final_state["results"]

  print("\n=== FINAL SYNTHESIZED OUTPUT ===")
  print(final_state["results"]["final.Synthesizer"])

  print("\n=== TEST PASSED ===")

# -------------------------
# Manual run support
# -------------------------
if __name__ == "__main__":
    test_workflow()
