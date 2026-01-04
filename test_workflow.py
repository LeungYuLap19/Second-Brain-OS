from src.managers.workflow_manager import WorkflowManager
from src.agents.registry import AGENT_REGISTRY
# from src.utils.helper import ingest_documents

def hybrid_agent_runner(agent_name: str, input_text: str) -> str:
  agent = AGENT_REGISTRY.get(agent_name)

  print(f"\n--- RUN: {agent_name} ---")

  if agent:
    return agent.run(input_text)

  # Fallback mock
  return f"[MOCK OUTPUT from {agent_name}]"

if __name__ == "__main__":
  # load vectordb
  # ingest_documents()
  manager = WorkflowManager(agent_runner=hybrid_agent_runner)
  print("Second Brain OS 🧠 (type 'exit' to quit)")
  while True:
    user_request = input("\nUser 🤡 > ").strip()
    if user_request.lower() == "exit":
      print("Exiting Second Brain OS. Goodbye!")
      break
    if user_request:
      final_state = manager.run(user_request)
      # print(final_state)
      last_step_key = sorted(final_state["results"].keys())[-1]
      print(f"Second Brain 🤖 > {final_state["results"][last_step_key]}")