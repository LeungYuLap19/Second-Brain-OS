import sqlite3
from typing import Optional, List, Dict, Any
from configs.paths import SQLITE_DB_FILE

def get_connection() -> sqlite3.Connection:
  """Return a sqlite3 Connection with Row factory set."""
  conn = sqlite3.connect(SQLITE_DB_FILE)
  conn.row_factory = sqlite3.Row
  return conn

def create_chatroom(chatroom_id: str, client_id: str) -> bool:
  """Create a chatroom row (or ensure it exists). If client_id is provided, set/update it.

  Returns True on success.
  """
  try:
    with get_connection() as conn:
      cur = conn.cursor()
      cur.execute(
        "INSERT OR IGNORE INTO chatrooms (id, client_id) VALUES (?, ?)",
        (chatroom_id, client_id),
      )
    return True
  except Exception:
    return False

def insert_message(chatroom_id: str, role: str, content: str, client_id: str) -> bool:
  """Insert a message into a chatroom. If the chatroom doesn't exist, create it first.

  If `client_id` is provided it will be stored on the chatroom row.
  Role must be 'user' or 'assistant'. Returns True on success.
  """
  if role not in ("user", "assistant"):
    return False

  try:
    create_chatroom(chatroom_id, client_id)

    with get_connection() as conn:
      cur = conn.cursor()
      cur.execute(
        "INSERT INTO messages (chatroom_id, role, content) VALUES (?, ?, ?)",
        (chatroom_id, role, content),
      )
    return True
  except Exception:
    return False

def delete_chatroom(chatroom_id: str) -> bool:
  """Delete a chatroom. Returns True if a row was deleted."""
  try:
    with get_connection() as conn:
      cur = conn.cursor()
      cur.execute("DELETE FROM chatrooms WHERE id = ?", (chatroom_id,))
      deleted = cur.rowcount
    return deleted > 0
  except Exception:
    return False

def delete_all_chatrooms(client_id: str) -> bool:
  """Delete all chatrooms belonging to a specific client_id.
  
  This will also delete all messages in those chatrooms due to foreign key constraints.
  Returns True on success, False on failure.
  """
  try:
    with get_connection() as conn:
      cur = conn.cursor()
      
      # Delete all chatrooms for this client
      # Messages will be automatically deleted due to foreign key CASCADE
      cur.execute("DELETE FROM chatrooms WHERE client_id = ?", (client_id,))
      deleted_count = cur.rowcount
    return True
  except Exception as e:
    print(f"Error deleting chatrooms for client_id {client_id}: {e}")
    return False

def retrieve_chatroom(chatroom_id: str) -> Optional[Dict[str, Any]]:
  """Retrieve a chatroom and its messages. Returns dict or False if not found."""
  try:
    with get_connection() as conn:
      cur = conn.cursor()
      cur.execute("SELECT id, client_id, created_at, updated_at FROM chatrooms WHERE id = ?", (chatroom_id,))
      row = cur.fetchone()

      if not row:
        return False

      cur.execute(
        "SELECT id, role, content, timestamp FROM messages WHERE chatroom_id = ? ORDER BY timestamp ASC",
        (chatroom_id,),
      )
      msgs = [
        {"id": m["id"], "role": m["role"], "content": m["content"], "timestamp": m["timestamp"]}
        for m in cur.fetchall()
      ]

    return {
      "id": row["id"],
      "client_id": row["client_id"],
      "created_at": row["created_at"],
      "updated_at": row["updated_at"],
      "messages": msgs,
    }
  except Exception:
    return False

def get_chat_history(client_id: str) -> Optional[List[Dict[str, Any]]]:
  """Return a list of chatrooms with their last updated timestamp and last message (if any).

  Return chatrooms that belong to that client.
  """
  try:
    with get_connection() as conn:
      cur = conn.cursor()
      cur.execute(
        """
        SELECT
          c.id AS id,
          c.updated_at AS updated_at,
          (
            SELECT content FROM messages WHERE chatroom_id = c.id ORDER BY timestamp DESC LIMIT 1
          ) AS last_message
        FROM chatrooms c
        WHERE c.client_id = ?
        ORDER BY c.updated_at DESC
        """,
        (client_id,)
      )
      rows = cur.fetchall()

    return [
      {
        "id": r["id"],
        "updated_at": r["updated_at"],
        "last_message": r["last_message"],
      }
      for r in rows
    ]
  except Exception as e:
    print(e)
    return False
