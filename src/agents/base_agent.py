# from configs.settings_loader import settings
from configs.settings_loader import settings
from langchain_ollama import ChatOllama
from ..tools.registry import TOOL_REGISTRY
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk

class BaseAgent:
  def __init__(self, name: str):
    self.name = name

    agent_config = settings.get_agent_model_config(name)
    system_prompt = settings.get_system_prompt(name)
    self.enable_streaming = agent_config.get("enable_streaming", False)

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

    # ALWAYS use create_agent — works perfectly with zero tools
    self.agent = create_agent(
      model=self.model,
      tools=self.tools,
      system_prompt=system_prompt,
    )

  def run(self, input_text: str):
    """
    Run the agent with the given input.
    
    :param input_text: User message
    :param thread_id: Unique identifier for the conversation/session (e.g., user ID)
    :return: Agent's response string
    """
    if self.enable_streaming:
        return self._run_streaming(input_text)
    else:
        return self._run_atomic(input_text)

  def _run_atomic(self, input_text: str) -> str:
    input_data = {"messages": [{"role": "user", "content": input_text}]}
    result = self.agent.invoke(input_data)
    return result["messages"][-1].content

  def _run_streaming(self, input_text: str):
    input_data = {"messages": [{"role": "user", "content": input_text}]}
    for stream_mode, data in self.agent.stream(
      input_data,
      stream_mode=["updates", "messages"]
    ):
      if stream_mode == "messages":
        token, _ = data
        if isinstance(token, AIMessageChunk) and token.content:
          yield token.content