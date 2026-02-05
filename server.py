# init env file
from dotenv import load_dotenv
from configs.paths import ENV_FILE
load_dotenv(ENV_FILE)

import os
import asyncio
import threading
import sqlite3
import sys
import io
import uvicorn
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict

# Import services
from src.services.vectordb import ingest_professor_document, delete_professor_document
from src.services.sqlite import insert_message, delete_chatroom, delete_all_chatrooms, retrieve_chatroom, get_chat_history

# Import LLM manager
from src.llm.llm_manager import LLMManager

logger = logging.getLogger("server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Second Brain OS Server")

# --- CORS Middleware ---
# Allow all origins by default; change to a restricted list in production
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# --- WebSocket connection manager ---
class ConnectionManager:
  def __init__(self):
    self.active_connections: Dict[str, WebSocket] = {}
    self.llm_managers: Dict[str, LLMManager] = {}
    self.client_chatrooms: Dict[str, set] = {}  # Track which chatrooms belong to which client

  async def connect(self, chatroom_id: str, websocket: WebSocket, client_id: str = None):
    await websocket.accept()
    self.active_connections[chatroom_id] = websocket
    self.llm_managers[chatroom_id] = LLMManager(chatroom_id)
    if client_id:
      if client_id not in self.client_chatrooms:
        self.client_chatrooms[client_id] = set()
      self.client_chatrooms[client_id].add(chatroom_id)
    logger.info(f"Chatroom connected: {chatroom_id}")

  def disconnect(self, chatroom_id: str, client_id: str = None):
    if chatroom_id in self.active_connections:
      del self.active_connections[chatroom_id]
    if chatroom_id in self.llm_managers:
      del self.llm_managers[chatroom_id]
    # Remove from client_chatrooms tracking
    if client_id and client_id in self.client_chatrooms:
      self.client_chatrooms[client_id].discard(chatroom_id)
      if not self.client_chatrooms[client_id]:  # Remove client if no chatrooms left
        del self.client_chatrooms[client_id]
    logger.info(f"Chatroom disconnected: {chatroom_id}")

  def get_llm_manager(self, chatroom_id: str) -> LLMManager:
    return self.llm_managers.get(chatroom_id)

  async def send_personal_message(self, message: str, chatroom_id: str):
    websocket = self.active_connections.get(chatroom_id)
    if websocket:
      await websocket.send_text(message)

  async def broadcast(self, message: str):
    for ws in list(self.active_connections.values()):
      try:
        await ws.send_text(message)
      except Exception:
        pass

  async def disconnect_chatroom(self, chatroom_id: str):
    """Disconnect a specific chatroom's websocket connection."""
    if chatroom_id in self.active_connections:
      websocket = self.active_connections[chatroom_id]
      try:
        await websocket.close(code=1000, reason="Chatroom deleted")
      except Exception:
        logger.exception(f"Error closing websocket for chatroom {chatroom_id}")
      # Find client_id for this chatroom
      client_id = None
      for cid, chatrooms in self.client_chatrooms.items():
        if chatroom_id in chatrooms:
          client_id = cid
          break
      self.disconnect(chatroom_id, client_id)

  async def disconnect_all_client_chatrooms(self, client_id: str):
    """Disconnect all chatroom websocket connections for a specific client."""
    if client_id in self.client_chatrooms:
      chatroom_ids = list(self.client_chatrooms[client_id])  # Copy to avoid modification during iteration
      for chatroom_id in chatroom_ids:
        if chatroom_id in self.active_connections:
          websocket = self.active_connections[chatroom_id]
          try:
            await websocket.close(code=1000, reason="All chatrooms deleted")
          except Exception:
            logger.exception(f"Error closing websocket for chatroom {chatroom_id}")
          self.disconnect(chatroom_id, client_id)

manager = ConnectionManager()

# --- Request models ---
class IngestRequest(BaseModel):
  filename: str

class DelDocRequest(BaseModel):
  filename: str

class InsertMessageRequest(BaseModel):
  chatroom_id: str
  role: str
  content: str
  client_id: str

class DeleteChatroomRequest(BaseModel):
  chatroom_id: str

class DeleteAllChatroomsRequest(BaseModel):
  client_id: str

# --- Routes ---
@app.get("/")
async def root():
  return {"status": "ok", "message": "Second Brain OS Server is running"}

@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
  """Trigger ingestion for a file present in INPUT_DATA_DIR by filename."""
  filename = request.filename
  logger.info(f"Ingest request received for: {filename}")
  success = ingest_professor_document(filename)

  if success:
    return JSONResponse({
      "success": True, 
      "message": f"Ingested {filename}"
    })

  raise HTTPException(
    status_code=404, 
    detail=f"File not found or ingestion failed: {filename}"
  )

@app.delete("/deldoc")
async def deldoc_endpoint(request: DelDocRequest):
  """Delete a document by filename from the RAG store and input dir. Use query param: /deldoc?filename=foo.pdf"""
  filename = request.filename
  
  if not filename:
    raise HTTPException(
      status_code=400, 
      detail="'filename' query parameter is required"
    )

  logger.info(f"Delete request received for: {filename}")
  success = delete_professor_document(filename)

  if success:
    return JSONResponse({
      "success": True, 
      "message": f"Deleted {filename}"
    })

  raise HTTPException(
    status_code=404, 
    detail=f"Document not found or deletion failed: {filename}"
  )

@app.delete("/delete_chatroom")
async def delete_chatroom_endpoint(request: DeleteChatroomRequest):
  chatroom_id = request.chatroom_id 
  logger.info(f"Delete chatroom request for chatroom: {chatroom_id}")
  success = delete_chatroom(chatroom_id)

  if success:
    # Disconnect the websocket connection for this chatroom
    await manager.disconnect_chatroom(chatroom_id)
    return JSONResponse({
      "success": True, 
      "message": f"Deleted chatroom: {chatroom_id}"
    })

  raise HTTPException(
    status_code=400, 
    detail=f"Deletion failed for chatroom: {chatroom_id}"
  )

@app.delete("/delete_all_chatrooms")
async def delete_all_chatrooms_endpoint(request: DeleteAllChatroomsRequest):
  client_id = request.client_id
  logger.info(f"Delete all chatrooms request for client: {client_id}")
  success = delete_all_chatrooms(client_id)

  if success:
    # Disconnect all websocket connections for this client's chatrooms
    await manager.disconnect_all_client_chatrooms(client_id)
    return JSONResponse({
      "success": True, 
      "message": f"Deleted all chatrooms for client: {client_id}"
    })

  raise HTTPException(
    status_code=400, 
    detail=f"Deletion failed for client: {client_id}"
  )

@app.get("/retrieve_chatroom/{chatroom_id}")
async def retrieve_chatroom_endpoint(chatroom_id: str):
  logger.info(f"Retrieving chatroom: {chatroom_id}")
  result = retrieve_chatroom(chatroom_id)

  if result:
    return JSONResponse({
      "success": True, 
      "message": f"Retrieved chatroom: {chatroom_id}",
      "data": result
    })
  
  raise HTTPException(
    status_code=404, 
    detail=f"Retrieval failed for chatroom: {chatroom_id}"
  )

@app.get("/chat_history/{client_id}")
async def chat_history_endpoint(client_id: str):
  logger.info(f"Retrieving chat history for client_id={client_id}")
  result = get_chat_history(client_id)

  if result is not False:
    return JSONResponse({"success": True, "data": result})

  raise HTTPException(
    status_code=500, 
    detail="Failed to retrieve chat history"
  )

# --- WebSocket endpoint ---
@app.websocket("/ws/{client_id}/{chatroom_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, chatroom_id: str):
  await manager.connect(chatroom_id, websocket, client_id)
  try:
    while True:
      data = await websocket.receive_text()
      logger.info(f"Received from chatroom {chatroom_id}: {data}")
      # Offload DB write to a thread so we don't block the event loop
      try:
        success = await asyncio.to_thread(insert_message, chatroom_id, "user", data, client_id)
        if not success:
          logger.warning(f"Failed to insert user message into chatroom: {chatroom_id}")
      except Exception:
        logger.exception("Exception while inserting user message into DB")

      # Stream LLM responses in a background thread so we don't block the event loop"
      loop = asyncio.get_running_loop()
      llm_manager_instance = manager.get_llm_manager(chatroom_id)
      if not llm_manager_instance:
        await manager.send_personal_message("Error: No LLM manager found for session", chatroom_id)
        continue

      def stream_and_send():
        try:
          # Notify client that the actual response is starting (separates from the processing ack)
          start_coro = manager.send_personal_message("<RESPONSE_START>", chatroom_id)
          try:
            asyncio.run_coroutine_threadsafe(start_coro, loop).result(timeout=5)
          except Exception:
            # If start marker fails to send, continue streaming anyway
            logger.exception("Failed to send RESPONSE_START marker")

          assistant_message = ''
          for chunk in llm_manager_instance.run(data):
            # Schedule send on the event loop thread-safely
            assistant_message += chunk
            coro = manager.send_personal_message(chunk, chatroom_id)
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            try:
              fut.result()
            except Exception:
              logger.exception("Failed to send LLM chunk to client")
          
          ins_success = insert_message(chatroom_id, "assistant", assistant_message, client_id)
          if not ins_success:
            logger.warning(f"Failed to insert assistant message into chatroom: {chatroom_id}")

          coro = manager.send_personal_message("<END_OF_STREAM>", chatroom_id)
          asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception:
          logger.exception("LLM streaming failed")

      loop.run_in_executor(None, stream_and_send)

  except WebSocketDisconnect:
    manager.disconnect(chatroom_id, client_id)
  except Exception as e:
    logger.exception("WebSocket error")
    manager.disconnect(chatroom_id, client_id)

if __name__ == "__main__":
  uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

