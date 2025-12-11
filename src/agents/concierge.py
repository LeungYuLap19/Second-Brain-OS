from .base_agent import BaseAgent

class ConciergeAgent(BaseAgent):
  def __init__(self):
    super().__init__("Concierge")

  def plan_travel(self, request: str):
    return self.run(request)

  def build_itinerary(self, info: str):
    return self.run(info)
