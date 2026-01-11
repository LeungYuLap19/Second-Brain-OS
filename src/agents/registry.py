from .professor import ProfessorAgent
from .responder import ResponderAgent
from .researcher import ResearcherAgent
from .communicator import CommunicatorAgent
from .accountant import AccountantAgent
from .secretary import SecretaryAgent

AGENT_REGISTRY = {
  "Professor": ProfessorAgent(), # perfect for pdf
  "Researcher": ResearcherAgent(), # perfect
  "Responder": ResponderAgent(), # perfect
  "Communicator": CommunicatorAgent(), # perfect
  "Accountant": AccountantAgent(), # need fix to do: fix accountant faking transactions (from memory, not facts)
  "Secretary": SecretaryAgent(), # need fix to do: date not accurate and too simple
}