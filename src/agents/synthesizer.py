from .base_agent import BaseAgent

class SynthesizerAgent(BaseAgent):
  def __init__(self):
    super().__init__("Synthesizer")

  def synthesize(self, aggregated_content: str):
    return self.run(aggregated_content)
