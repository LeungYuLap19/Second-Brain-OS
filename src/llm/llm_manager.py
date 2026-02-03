import socket
from collections import deque
from datetime import datetime
from typing import Optional
import logging
from configs.settings_loader import settings
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk

from src.tools.tavily import tavily_search_api, tavily_extract_content
from src.tools.doc_tools import search_documents
from src.tools.gmail import get_emails, gmail_send_message
from src.tools.calendar import search_calendar_events, create_calendar_event, update_calendar_event, delete_calendar_event
from src.services.sqlite import retrieve_chatroom

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

  def __init__(self, chatroom_id: Optional[str] = None, short_term_memory_size: int = 5):
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

    # If a chatroom_id is provided, attempt to load recent turns from persistent storage
    if chatroom_id:
      try:
        loaded = self._load_existing_memory(chatroom_id)
        if loaded:
          logger.info(f"Loaded existing short-term memory for chatroom {chatroom_id} ({len(self.short_memory)} turns)")
        else:
          logger.info(f"No existing short-term memory found for chatroom {chatroom_id}")
      except Exception:
        logger.exception("Failed to load existing memory for chatroom %s", chatroom_id)

  def _is_connected(self):
    try:
      socket.create_connection(("www.google.com", 443), timeout=3)
      return True
    except OSError:
      return False

  def _load_existing_memory(self, chatroom_id: str):
    """Load recent turns from persistent storage into short-term memory.

    This reads messages from the chatroom and aggregates them into user/assistant
    turns. Existing short-term memory is preserved; the oldest turns will be
    trimmed automatically by the deque's maxlen.
    Returns True if any turns were loaded, False otherwise.
    """
    try:
      room = retrieve_chatroom(chatroom_id)
      if not room or "messages" not in room:
        return False

      msgs = room.get("messages", [])

      # Build turns from chronological messages
      pending_user = None
      pending_user_ts = None
      loaded_any = False

      for m in msgs:
        role = m.get("role")
        content = m.get("content", "")
        ts = m.get("timestamp")

        if role == "user":
          # Start/replace the pending user message
          pending_user = content
          pending_user_ts = ts
        elif role == "assistant":
          # If we have a pending user, pair them; otherwise record assistant-only turn
          if pending_user is not None:
            self._add_turn(pending_user, content, pending_user_ts or ts)
            pending_user = None
            pending_user_ts = None
            loaded_any = True
          else:
            # Assistant response with no preceding user; add as anonymous user turn
            self._add_turn("", content, ts)
            loaded_any = True

      # If the last message is a user message without assistant response, include it as a partial turn
      if pending_user is not None:
        self._add_turn(pending_user, "", pending_user_ts)
        loaded_any = True

      return loaded_any
    except Exception:
      # Don't let loading memory break the LLM manager construction
      return False

  def _add_turn(self, user_text: str, assistant_text: str, timestamp: str = None):
    """Append a user-assistant turn to short-term memory.

    If `timestamp` is not provided, the current UTC ISO timestamp with Z suffix is used.
    """
    if timestamp is None:
      timestamp = datetime.utcnow().isoformat() + "Z"
    self.short_memory.append({"user": user_text, "assistant": assistant_text, "timestamp": timestamp})

  def _get_memory_messages(self):
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
    messages = self._get_memory_messages() + [{"role": "user", "content": input_text}]
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
        self._add_turn(input_text, full_response)
      except Exception:
        # Don't let memory saving break streaming
        pass