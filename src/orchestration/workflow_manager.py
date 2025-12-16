from __future__ import annotations
from typing import Callable
from datetime import datetime, timezone
from langgraph.graph import StateGraph, START, END
from src.schemas.task_state import TaskState, TaskStatus
from src.schemas.data_models import OrchestratorPlan

class WorkflowManager:
  """
  LangGraph-based workflow executor.

  Guarantees:
  - All tasks from OrchestratorPlan run first
  - Synthesizer ALWAYS runs last
  """

  def __init__(self, agent_runner: Callable[[str, str], str]):
    """
    the function that runs an agent
    agent_runner(agent_name: str, input_text: str) -> output_text: str

    AGENT_REGISTRY = {
      "Professor": ProfessorAgent(),
      "Researcher": ResearcherAgent(),
      "Note Taker": NoteTakerAgent(),
      ...
    }

    def agent_runner(agent_name: str, input_text: str) -> str:
      agent = AGENT_REGISTRY.get(agent_name)
      if not agent:
        raise ValueError(f"Unknown agent: {agent_name}")
    """
    self.agent_runner = agent_runner

  # -------------------------
  # Graph construction
  # ------------------------- 
  def build_graph(self, plan: OrchestratorPlan) -> StateGraph:
    """
    define the State of the graph
    """
    graph = StateGraph(TaskState)

    for task in plan.tasks:
      graph.add_node(
        self._task_node_name(task.step, task.agent),
        self._make_task_node(task.step)
      )

    graph.add_node("synthesizer", self._make_synthesizer_node())

    # Wire task nodes sequentially
    for index, task in enumerate(plan.tasks):
      current = self._task_node_name(task.step, task.agent)

      if index == len(plan.tasks) - 1:
        graph.add_edge(current, "synthesizer")
      else:
        next_task = plan.tasks[index + 1]
        graph.add_edge(current, self._task_node_name(next_task.step, next_task.agent))

    # Start with first agent
    graph.add_edge(START, self._task_node_name(plan.tasks[0].step, plan.tasks[0].agent))
    # Terminate with synthesizer
    graph.add_edge("synthesizer", END)
    return graph

  # -------------------------
  # Node factories
  # -------------------------
  def _make_task_node(self, step: int):
    """
    define node
    node have to update TaskState
    """
    def node(state: TaskState) -> TaskState:
      task = state.tasks[step]

      if task.status == TaskStatus.COMPLETED:
        return state

      try:
        state.mark_running(step)
        input_text = self._resolve_inputs(state, task.inputs, task.instruction)
        output = self.agent_runner(task.agent, input_text)

        output_key = f"step{step}.{task.agent}"
        state.mark_completed(step, output_key, output)

      except Exception as e:
        state.mark_failed(step, str(e))

      return state
    
    return node
  
  def _make_synthesizer_node(self):
    """
    define node
    node have to update TaskState
    """
    def node(state: TaskState) -> TaskState:
      ordered_outputs = []
      for step in sorted(state.tasks.keys()):
        task = state.tasks[step]
        if task.status == TaskStatus.COMPLETED and task.output:
          ordered_outputs.append(task.output)
          
      combined_context = "\n\n".join(ordered_outputs)
      input_text = (
        f"Instruction: \nSynthesize all task outputs into a final user-facing response.\n\n"
        f"Context: \n{combined_context}"
      )

      try:
        final_output = self.agent_runner("Synthesizer", input_text)
      except Exception as e:
        final_output = str(e)

      state.results["final.Synthesizer"] = final_output
      state.updated_at = datetime.now(timezone.utc)
      return state
    
    return node

  # -------------------------
  # Input resolution
  # -------------------------
  def _resolve_inputs(self, state: TaskState, input_keys: list[str], instruction: str) -> str:
    """
    input_keys: state key indicates which existing value to use
    merge context and instruction to final text input
    """
    resolved = []

    for key in input_keys:
      if key == "user_request":
        resolved.append(state.user_request)
      elif key in state.results:
        resolved.append(state.results[key])
      else:
        raise ValueError(f"Missing input: {key}")

    context = "\n\n".join(resolved)

    return (
      f"Instruction: \n{instruction}\n\n"
      f"Context:\n{context}"
    )

  @staticmethod
  def _task_node_name(step: int, agent_name: str) -> str:
    return f"step{step}.{agent_name}"