from src.managers.workflow_manager import WorkflowManager
from src.agents.registry import AGENT_REGISTRY
from src.utils.helper import ingest_professor_documents, clear_memory_vdb
import shutil

def hybrid_agent_runner(agent_name: str, input_text: str, stream: bool = False):
  agent = AGENT_REGISTRY.get(agent_name)

  if not agent:
    mock_output = f"[MOCK OUTPUT from {agent_name}]"
    if stream:
      for char in mock_output:
        yield char
    else:
      return mock_output

  result = agent.run(input_text)

  if stream:
    for chunk in result:
      yield chunk
  else:
    if hasattr(result, '__iter__') and not isinstance(result, str):
      return ''.join(result) 
    else:
      return result 

if __name__ == "__main__":
  clear_memory_vdb()
  ingest_professor_documents()
  manager = WorkflowManager(agent_runner=hybrid_agent_runner)
  print("Second Brain OS 🧠 (type 'exit' to quit)\n")
  while True:
    print("| User 🤡 >", end=" ")
    user_request = input().strip()
    print("|", "-" * (shutil.get_terminal_size().columns - 2))
    if user_request.lower() == "exit":
      print("Exiting Second Brain OS. Goodbye!")
      break 
    if user_request:
      manager.run(user_request)