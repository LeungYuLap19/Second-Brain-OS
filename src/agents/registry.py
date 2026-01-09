from .professor import ProfessorAgent
from .responder import ResponderAgent
from .researcher import ResearcherAgent
from .communicator import CommunicatorAgent
from .accountant import AccountantAgent
from .secretary import SecretaryAgent

AGENT_REGISTRY = {
  "Professor": ProfessorAgent(),
  "Researcher": ResearcherAgent(),
  "Responder": ResponderAgent(),
  "Communicator": CommunicatorAgent(),
  "Accountant": AccountantAgent(),
  "Secretary": SecretaryAgent(),
}