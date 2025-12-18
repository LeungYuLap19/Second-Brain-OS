# from configs.settings_loader import settings
from configs.settings_loader import settings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from ..tools.registry import TOOL_REGISTRY
from langchain.agents import create_agent

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

    if self.tools:
      self.agent = create_agent(
        model=self.model,
        tools=self.tools,
        system_prompt=system_prompt
      )
    else:
      self.prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
      ])
      self.chain = self.prompt | self.model


  def run(self, input_text: str):
    if hasattr(self, "agent"):
      result = self.agent.invoke(
        {"messages": [{"role": "user", "content": input_text}]}
      ) 
      messages = result["messages"]
      final_message = messages[-1]

      return final_message.content
    else:
      return self.chain.invoke({"input": input_text}).content