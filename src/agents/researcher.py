from .base_agent import BaseAgent

class ResearcherAgent(BaseAgent):
  def __init__(self):
    super().__init__("Researcher")

  def search(self, query: str):
    return self.run(query)
