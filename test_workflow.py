from src.managers.workflow_manager import WorkflowManager
from src.agents.registry import AGENT_REGISTRY
from src.utils.helper import ingest_professor_documents, clear_memory_vdb

def hybrid_agent_runner(agent_name: str, input_text: str) -> str:
  agent = AGENT_REGISTRY.get(agent_name)

  print(f"\n--- RUN: {agent_name} ---")

  if agent:
    return agent.run(input_text)

  # Fallback mock
  return f"[MOCK OUTPUT from {agent_name}]"

if __name__ == "__main__":
  # load vectordb
  # ingest_professor_documents()
  clear_memory_vdb()
  manager = WorkflowManager(agent_runner=hybrid_agent_runner)
  print("Second Brain OS 🧠 (type 'exit' to quit)")
  while True:
    user_request = input("\nUser 🤡 > ").strip()
    if user_request.lower() == "exit":
      print("Exiting Second Brain OS. Goodbye!")
      break
    if user_request:
      final_state = manager.run(user_request)
      last_step = max(final_state["tasks"].keys())
      last_task = final_state["tasks"][last_step]
      final_output = last_task.output
      print(f"Second Brain 🤖 > {final_output}")