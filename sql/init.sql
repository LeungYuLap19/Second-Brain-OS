CREATE TABLE IF NOT EXISTS chatrooms (
  id TEXT PRIMARY KEY,
  client_id TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chatroom_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (chatroom_id) REFERENCES chatrooms(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chatrooms_client_id ON chatrooms(client_id);
CREATE INDEX IF NOT EXISTS idx_messages_chatroom_id ON messages(chatroom_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);

-- Create trigger to update chatroom's updated_at timestamp when new message is added
CREATE TRIGGER IF NOT EXISTS update_chatroom_timestamp 
AFTER INSERT ON messages
BEGIN
  UPDATE chatrooms 
  SET updated_at = CURRENT_TIMESTAMP 
  WHERE id = NEW.chatroom_id;
END;