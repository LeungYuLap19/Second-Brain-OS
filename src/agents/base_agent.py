# from configs.settings_loader import settings
from configs.settings_loader import settings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from ..tools.registry import TOOL_REGISTRY
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

class BaseAgent:
  def __init__(self, name: str):
    self.name = name

    agent_config = settings.get_agent_model_config(name)
    system_prompt = settings.get_system_prompt(name)

    self.model = ChatOllama(
      model=agent_config["model"],
      temperature=agent_config.get("temperature", 0.7),
      num_ctx=agent_config.get("num_ctx", 4096),
      num_predict=agent_config.get("num_predict"),
      base_url=settings.get_base_url(),
      format=agent_config.get("format", ""),
    )

    tool_names = agent_config.get("tools", [])
    self.tools = [
      TOOL_REGISTRY[name]
      for name in tool_names
      if name in TOOL_REGISTRY
    ]

    # Create a checkpointer for short-term memory
    # In-memory for development
    self.checkpointer = InMemorySaver()        

    # ALWAYS use create_agent — works perfectly with zero tools
    self.agent = create_agent(
      model=self.model,
      tools=self.tools,
      system_prompt=system_prompt,
      checkpointer=self.checkpointer,
    )

  def run(self, input_text: str, thread_id: str = "default"):
    """
    Run the agent with the given input.
    
    :param input_text: User message
    :param thread_id: Unique identifier for the conversation/session (e.g., user ID)
    :return: Agent's response string
    """
    config = {"configurable": {"thread_id": thread_id}}

    result = self.agent.invoke(
      {"messages": [{"role": "user", "content": input_text}]},
      config=config,
    )

    final_message = result["messages"][-1]
    return final_message.content