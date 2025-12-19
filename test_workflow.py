from src.agents.orchestrator import OrchestratorAgent
from src.schemas.task_state import TaskState
from src.schemas.data_models import OrchestratorPlan
from src.orchestration.workflow_manager import WorkflowManager
from src.agents.professor import ProfessorAgent
from src.agents.responder import ResponderAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.researcher import ResearcherAgent
from src.tools.doc_ingest import ingest_documents

AGENT_REGISTRY = {
  "Professor": ProfessorAgent(),
  "Researcher": ResearcherAgent(),
  "Responder": ResponderAgent(),
  "Synthesizer": SynthesizerAgent(),
}

# -------------------------
# Hybrid agent runner
# -------------------------
def hybrid_agent_runner(agent_name: str, input_text: str) -> str:
  """
  Uses real agents if registered.
  Falls back to mock for unimplemented agents.
  """
  agent = AGENT_REGISTRY.get(agent_name)

  print(f"\n--- RUN: {agent_name} ---")
  print(input_text)
  print("-------------------------")

  if agent:
    return agent.run(input_text)

  # Fallback mock
  return f"[MOCK OUTPUT from {agent_name}]"

# -------------------------
# Workflow test
# -------------------------
def run_workflow(user_request: str):
  orch = OrchestratorAgent()
  plan = orch.run(user_request)
  plan = OrchestratorPlan.model_validate(plan)

  print("\n=== ORCHESTRATOR PLAN ===")
  for task in plan.tasks:
    print(task)

  initial_state = TaskState()
  initial_state.init_from_plan(plan=plan, user_request=user_request)

  manager = WorkflowManager(agent_runner=hybrid_agent_runner)
  graph = manager.build_graph(plan)
  app = graph.compile()
  final_state = app.invoke(initial_state)

  for step, task in final_state["tasks"].items():
    if task.status.name == "COMPLETED":
      print("\n=== FINAL OUTPUT ===")
      last_step_key = sorted(final_state["results"].keys())[-1]
      print(final_state["results"][last_step_key])
      print("\n=== WORKFLOW COMPLETED ===")
    else:
      print("\n=== WORKFLOW FAILED ===")
      print(final_state)

  # assert "final.Synthesizer" in final_state["results"]

# -------------------------
# Terminal input support
# -------------------------
if __name__ == "__main__":
  # load vectordb
  ingest_documents()
  print("Second Brain OS (type 'exit' to quit)")
  while True:
    user_request = input("> ").strip()
    if user_request.lower() == "exit":
      print("Exiting Second Brain OS. Goodbye!")
      break
    if user_request:
  # user_request = ("")
      run_workflow(user_request)