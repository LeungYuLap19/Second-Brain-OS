from .base_agent import BaseAgent

class ResponderAgent(BaseAgent):
  def __init__(self):
    super().__init__("Responder")

  def respond(self, text: str):
    return self.run(text)