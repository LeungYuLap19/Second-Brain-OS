import socket
from collections import deque
from datetime import datetime
from configs.settings_loader import settings
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk

from src.tools.tavily import tavily_search_api, tavily_extract_content
from src.tools.doc_tools import search_documents
from src.tools.gmail import get_emails, gmail_send_message
from src.tools.calendar import search_calendar_events, create_calendar_event, update_calendar_event, delete_calendar_event

class LLMManager:
  """
  Central manager for Large Language Model (LLM) operations in Second Brain OS.
    
  This class handles:
  - LLM initialization with automatic cloud/local fallback
  - Agent creation with integrated tools
  - Streaming response generation
  - Simple short-term memory (recent N turns)
  - Internet connectivity detection for model selection
  """

  def __init__(self, short_term_memory_size: int = 5):
    llm_model_config = settings.get_llm_model_config()
    system_prompt = settings.get_system_prompt()

    self.model = ChatOllama(
      base_url=settings.get_base_url(),
      model=llm_model_config["model"] if self._is_connected() else llm_model_config["local_model"],
      temperature=llm_model_config.get("temperature"),
      num_ctx=llm_model_config.get("num_ctx"),
      num_predict=llm_model_config.get("num_predict"),
      streaming=True
    )

    self.tools = [
      search_documents,
      tavily_search_api,
      tavily_extract_content,
      get_emails,
      gmail_send_message,
      search_calendar_events,
      create_calendar_event,
      update_calendar_event,
      delete_calendar_event
    ]

    self.llm_agent = create_agent(
      model=self.model,
      tools=self.tools,
      system_prompt=system_prompt
    )

    # Short-term memory holds recent conversation turns as dicts: {'user':..., 'assistant':...}
    self.short_memory = deque(maxlen=short_term_memory_size)

  def _is_connected(self):
    try:
      socket.create_connection(("www.google.com", 443), timeout=3)
      return True
    except OSError:
      return False

  def add_turn(self, user_text: str, assistant_text: str, timestamp: str = None):
    """Append a user-assistant turn to short-term memory.

    If `timestamp` is not provided, the current UTC ISO timestamp with Z suffix is used.
    """
    if timestamp is None:
      timestamp = datetime.utcnow().isoformat() + "Z"
    self.short_memory.append({"user": user_text, "assistant": assistant_text, "timestamp": timestamp})

  def get_memory_messages(self):
    """Return short-term memory formatted as a list of messages suitable for input_data.

    Each message includes an ISO UTC timestamp prefix so the model can reason about recency.
    """
    msgs = []
    for turn in self.short_memory:
      ts = turn.get("timestamp")
      user_text = turn.get("user", "")
      assistant_text = turn.get("assistant", "")
      if ts:
        user_content = f"[{ts}] User: {user_text}"
        assistant_content = f"[{ts}] Assistant: {assistant_text}"
      else:
        user_content = user_text
        assistant_content = assistant_text
      msgs.append({"role": "user", "content": user_content})
      msgs.append({"role": "assistant", "content": assistant_content})
    return msgs

  def run(self, input_text: str):
    """Stream tokens from the agent. Includes short-term memory in the input messages and
    saves the final assistant response back into short-term memory when finished.
    """
    # include short-term memory turns before the current user message
    messages = self.get_memory_messages() + [{"role": "user", "content": input_text}]
    input_data = {"messages": messages}

    full_response_parts = []
    for stream_mode, data in self.llm_agent.stream(
      input_data,
      stream_mode=["updates", "messages"]
    ):
      if stream_mode == "messages":
        token, _ = data
        if isinstance(token, AIMessageChunk) and token.content:
          chunk = token.content
          full_response_parts.append(chunk)
          yield chunk

    # After streaming is complete, save the full response to short-term memory
    full_response = "".join(full_response_parts).strip()
    if full_response:
      try:
        self.add_turn(input_text, full_response)
      except Exception:
        # Don't let memory saving break streaming
        pass