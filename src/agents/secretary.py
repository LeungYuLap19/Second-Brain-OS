from .base_agent import BaseAgent

class SecretaryAgent(BaseAgent):
  def __init__(self):
    super().__init__("Secretary")

  def schedule(self, text: str):
    return self.run(text)

  def summarize_agenda(self, agenda: str):
    return self.run(agenda)
