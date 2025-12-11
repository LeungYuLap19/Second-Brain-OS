from .base_agent import BaseAgent

class AccountantAgent(BaseAgent):
  def __init__(self):
    super().__init__("Accountant")

  def extract_expense(self, text: str):
    return self.run(text)

  def validate(self, content: str):
    return self.run(content)
