from __future__ import annotations
from typing import Callable
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from src.schemas.task_state import TaskState, TaskStatus
from src.schemas.data_models import OrchestratorPlan
from src.agents.orchestrator import OrchestratorAgent
from src.agents.distiller import DistillerAgent
from src.utils.helper import ingest_memory_texts
import traceback
import json
from datetime import datetime

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
    """
    self.agent_runner = agent_runner
    self.checkpointer = InMemorySaver()
    self.thread_id = "default"
    self.app = None

  # -------------------------
  # Public API
  # ------------------------- 
  def run(self, user_request: str) -> TaskState:
    """
    High-level workflow execution entrypoint.
    """
    orchestrator = OrchestratorAgent()

    memory_context = self._get_state_memory(limit=5)
    orchestrator_input = (
      f"User request:\n{user_request}\n\n"
      f"Memory (JSON, read-only, do NOT use as execution inputs):\n{memory_context}"
      f"""
### CRITICAL RULES (READ THIS LAST):
* Memory is strictly read-only.
* Memory may only inform agent selection and instructions.
* Each plan executes in a fresh state.
* Step numbering restarts from 1 every turn.
      """
    )

    raw_plan = orchestrator.run(orchestrator_input)
    plan = OrchestratorPlan.model_validate(raw_plan)

    state = TaskState()
    state.init_from_plan(plan=plan, user_request=user_request)

    self.app = self._compile_with_memory(plan)

    final_state = self.app.invoke(
      state,
      {"configurable": {"thread_id": self.thread_id}}
    )

    return final_state

  # -------------------------
  # Graph construction
  # ------------------------- 
  def _build_graph(self, plan: OrchestratorPlan) -> StateGraph:
    # Define global workflow state for the graph
    graph = StateGraph(TaskState)

    # Add nodes
    for task in plan.tasks:
      graph.add_node(
        self._task_node_name(task.step, task.agent), 
        self._make_task_node(task.step)
      )

    # Comment out synthesizer for now
    # graph.add_node("synthesizer", self._make_synthesizer_node())

    # Add edges
    for index, task in enumerate(plan.tasks):
      current = self._task_node_name(task.step, task.agent)

      if index < len(plan.tasks) - 1:
        next_task = plan.tasks[index + 1]
        graph.add_edge(current, self._task_node_name(next_task.step, next_task.agent))
      # else:
      #     graph.add_edge(current, "synthesizer")  # disabled

    # Add START and END
    graph.add_edge(START, self._task_node_name(plan.tasks[0].step, plan.tasks[0].agent))
    # graph.add_edge("synthesizer", END)  # disabled

    # Connect last task directly to END
    last_task = plan.tasks[-1]
    graph.add_edge(self._task_node_name(last_task.step, last_task.agent), END)

    return graph
  
  def _compile_with_memory(self, plan: OrchestratorPlan) -> StateGraph:
    graph = self._build_graph(plan)
    return graph.compile(checkpointer=self.checkpointer)

  # -------------------------
  # Node factories
  # -------------------------
  def _make_task_node(self, step: int):
    def node(state: TaskState) -> TaskState:
      # current task 
      task = state.tasks[step]

      # return state upon task completion
      if task.status == TaskStatus.COMPLETED:
        return state

      try:
        # run current task
        state.mark_running(step)
        input_text = self._resolve_inputs(task.instruction)
        output = self.agent_runner(task.agent, input_text)

        # generate summary
        distiller = DistillerAgent()
        summary = distiller.run(output)

        # mark current task complete
        state.mark_completed(step, summary, output)

        # ingest memory
        ingest_memory_texts(
          texts=[output],
          metadatas=[{
            "agent": task.agent,
            "step": task.step,
            "user_request": state.user_request,
          }]
        )

      except Exception as e:
        tb = traceback.format_exc()
        print("\n🔥 TASK FAILED TRACEBACK 🔥")
        print(tb)
        print("🔥 END TRACEBACK 🔥\n")
        # mark current task failed
        state.mark_failed(step, str(e))

      return state
    
    return node

  # -------------------------
  # Input resolution
  # -------------------------
  def _resolve_inputs(self, instruction: str) -> str:
    memory_context = self._get_state_memory()

    return (
      f"Instruction: \n{instruction}\n\n"
      f"State Memory (JSON, read-only):\n{memory_context}\n\n"
    )

  def _get_state_memory(self, limit: int = 5) -> str:
    if not self.app:
      return "{}"

    config = {"configurable": {"thread_id": self.thread_id}}
    history = list(self.app.get_state_history(config))

    memory_entries = []
    seen_requests = set()

    for snapshot in history:
      if not snapshot.values:
        continue

      user_request = snapshot.values.get("user_request")
      if not user_request or user_request in seen_requests:
        continue

      entry = {"user_request": user_request}

      tasks_summary = []
      tasks = snapshot.values.get("tasks", {})
      for task in tasks.values():
        if task.status == TaskStatus.COMPLETED and task.output:
          tasks_summary.append({
            "agent": task.agent,
            "summary": task.summary
          })

      if tasks_summary:
        entry["task_outputs"] = tasks_summary

      memory_entries.append(entry)
      seen_requests.add(user_request)

      if len(memory_entries) >= limit:
        break

    if not memory_entries:
        base_context = {}
    else:
        base_context = {"memory": list(reversed(memory_entries))}

    # === INJECT CURRENT CONTEXT ===
    # You can get these from your session/user context
    current_datetime = datetime.now().astimezone()  # or use a fixed one for consistency
    current_location = "Hong Kong, HK"  # pull from user profile or IP

    # Format cleanly for the LLM
    context_injection = {
      "current_datetime": current_datetime.strftime("%Y-%m-%d %H:%M:%S %Z"),
      "current_date": current_datetime.strftime("%Y-%m-%d"),
      "current_time": current_datetime.strftime("%H:%M:%S %Z"),
      "current_location": current_location,
      "current_timezone": str(current_datetime.tzinfo),
      "note": "This is the real-time context. Use it to interpret relative dates like 'today', 'this week', 'last month', etc."
    }

    # Merge: context first, then historical memory
    final_memory = {**context_injection, **base_context}

    return json.dumps(final_memory, indent=2)

  @staticmethod
  def _task_node_name(step: int, agent_name: str) -> str:
    return f"step{step}.{agent_name}"
  




  # -------------------------
  # archived
  # -------------------------

  # def _make_synthesizer_node(self):
  #   """
  #   Optional synthesizer node (currently unused).
  #   """
  #   def node(state: TaskState) -> TaskState:
  #     # Extract all task outputs
  #     ordered_outputs = []
  #     for step in sorted(state.tasks.keys()):
  #       task = state.tasks[step]
  #       if task.status == TaskStatus.COMPLETED and task.output:
  #         ordered_outputs.append(task.output)
          
  #     # Combine all task outputs
  #     combined_context = "\n\n".join(ordered_outputs)
  #     input_text = (
  #       f"Instruction: \nSynthesize all task outputs into a final user-facing response.\n\n"
  #       f"Context: \n{combined_context}"
  #     )

  #     try:
  #       # Run synthesizer
  #       final_output = self.agent_runner("Synthesizer", input_text)
  #     except Exception as e:
  #       final_output = str(e)

  #     # Manually add synthesizer result
  #     state.results["final.Synthesizer"] = final_output
  #     state.updated_at = datetime.now(timezone.utc)
  #     return state
    
  #   return node