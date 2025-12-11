from .base_agent import BaseAgent

class CommunicatorAgent(BaseAgent):
  def __init__(self):
    super().__init__("Communicator")

  def summarize_email(self, email_text: str):
    return self.run(email_text)

  def draft_reply(self, context: str):
    return self.run(context)
