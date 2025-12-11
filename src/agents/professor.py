from .base_agent import BaseAgent

class ProfessorAgent(BaseAgent):
  def __init__(self):
    super().__init__("Professor")

  def summarize(self, text: str):
    return self.run(text)

  def extract(self, text: str):
    return self.run(text)
