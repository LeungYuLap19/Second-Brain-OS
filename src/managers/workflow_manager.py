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
from halo import Halo
import shutil

class WorkflowManager:
  """
  LangGraph-based workflow executor.
  """

  def __init__(self, agent_runner: Callable[[str, str, bool]]):
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
    self.spinner = Halo(text='Second Brain 🤖 > I am thinking! Patient!', spinner='dots', color=None)
    self.spinner.start()

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
    num_tasks = len(plan.tasks)

    # Add nodes
    for i, task in enumerate(plan.tasks):
      is_last = (i == num_tasks - 1)
      node_name = self._task_node_name(task.step, task.agent)
      graph.add_node(
        node_name,
        self._make_task_node(task.step, is_last_task=is_last)
      )

    # Add edges
    for i in range(num_tasks - 1):
      current = self._task_node_name(plan.tasks[i].step, plan.tasks[i].agent)
      next_node = self._task_node_name(plan.tasks[i+1].step, plan.tasks[i+1].agent)
      graph.add_edge(current, next_node)

    # Add START and END
    graph.add_edge(START, self._task_node_name(plan.tasks[0].step, plan.tasks[0].agent))

    # Connect last task directly to END
    graph.add_edge(self._task_node_name(plan.tasks[-1].step, plan.tasks[-1].agent), END)

    return graph
  
  def _compile_with_memory(self, plan: OrchestratorPlan) -> StateGraph:
    graph = self._build_graph(plan)
    return graph.compile(checkpointer=self.checkpointer)

  # -------------------------
  # Node factories
  # -------------------------
  def _make_task_node(self, step: int, is_last_task: bool = False):
    def node(state: TaskState) -> TaskState:
      # current task 
      task = state.tasks[step]

      # return state upon task completion
      if task.status == TaskStatus.COMPLETED:
        return state

      try:
        # run current task
        should_stream = is_last_task
        state.mark_running(step)
        input_text = self._resolve_inputs(task.instruction)
        output = self.agent_runner(task.agent, input_text, should_stream)

        if should_stream:
          full_output = ""
          for index, chunk in enumerate(output):
            if index == 0:
              self.spinner.stop()
              print(f"| Second Brain 🤖 ({task.agent}) >", end=" ")
            print(chunk, end="", flush=True)
            full_output += chunk
          print()
          print("|", "-" * (shutil.get_terminal_size().columns - 2))
        else:
          full_output = output

        self.spinner = Halo(text='Second Brain 🤖 > I am memorizing!', spinner='dots', color=None)
        self.spinner.start()
        
        # generate summary
        distiller = DistillerAgent()
        summary = distiller.run(full_output)

        # mark current task complete
        state.mark_completed(step, summary, full_output)

        # ingest memory
        ingest_memory_texts(
          text=full_output,
          metadata={
            "agent": task.agent,
            "step": task.step,
            "user_request": state.user_request,
          }
        )
        self.spinner.stop()

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
        base_context = {"state_memory": list(reversed(memory_entries))}

    # === INJECT CURRENT CONTEXT ===
    # You can get these from your session/user context
    current_datetime = datetime.now().astimezone()  # or use a fixed one for consistency
    current_location = "Hong Kong, HK"  # pull from user profile or IP

    # Format cleanly for the LLM
    context_injection = {
      "current user": "Jimmy",
      "current_datetime": current_datetime.strftime("%A %Y-%m-%d %H:%M:%S %Z"),
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