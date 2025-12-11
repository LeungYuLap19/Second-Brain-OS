from .base_agent import BaseAgent

class NoteTakerAgent(BaseAgent):
  def __init__(self):
    super().__init__("NoteTaker")

  def format_notes(self, content: str):
    return self.run(content)
