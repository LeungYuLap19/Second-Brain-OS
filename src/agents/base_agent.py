from configs.settings_loader import settings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class BaseAgent:
  def __init__(self, name: str):
    self.name = name

    agent_config = settings.get_agent_model_config(name)
    system_prompt = settings.get_system_prompt(name)

    self.model = ChatOllama(
      model=agent_config["model"],
      temperature=agent_config.get("temperature", 0.7),
      num_ctx=agent_config.get("num_ctx", 4096),
      base_url=agent_config.get("base_url", "http://localhost:11434"),
      format=agent_config.get("format", "")
    )

    self.prompt = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("human", "{input}")
    ])

  def run(self, input_text: str):

    
    chain = self.prompt | self.model
    return chain.invoke({"input": input_text}).content

