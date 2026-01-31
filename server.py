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

# Import helper functions for ingestion and deletion
from src.utils.helpers import ingest_professor_document, delete_professor_document

# Import LLM manager
from src.llm.llm_manager import LLMManager

logger = logging.getLogger("server")
logging.basicConfig(level=logging.INFO)

# Instantiate the LLM manager once (shared)
llm_manager = LLMManager()

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

  async def connect(self, client_id: str, websocket: WebSocket):
    await websocket.accept()
    self.active_connections[client_id] = websocket
    logger.info(f"Client connected: {client_id}")

  def disconnect(self, client_id: str):
    if client_id in self.active_connections:
      del self.active_connections[client_id]
      logger.info(f"Client disconnected: {client_id}")

  async def send_personal_message(self, message: str, client_id: str):
    websocket = self.active_connections.get(client_id)
    if websocket:
      await websocket.send_text(message)

  async def broadcast(self, message: str):
    for ws in list(self.active_connections.values()):
      try:
        await ws.send_text(message)
      except Exception:
        pass

manager = ConnectionManager()

# --- Request models ---
class IngestRequest(BaseModel):
  filename: str

class DelDocRequest(BaseModel):
  filename: str

# --- Routes ---
@app.get("/")
async def root():
  return {"status": "ok", "message": "Second Brain OS Server is running"}

@app.post("/ingest")
async def ingest(request: IngestRequest):
  """Trigger ingestion for a file present in INPUT_DATA_DIR by filename."""
  filename = request.filename
  logger.info(f"Ingest request received for: {filename}")
  success = ingest_professor_document(filename)
  if success:
    return JSONResponse({"success": True, "message": f"Ingested {filename}"})
  raise HTTPException(status_code=404, detail=f"File not found or ingestion failed: {filename}")

@app.delete("/deldoc")
async def deldoc(filename: str = ""):
  """Delete a document by filename from the RAG store and input dir. Use query param: /deldoc?filename=foo.pdf"""
  if not filename:
    raise HTTPException(status_code=400, detail="'filename' query parameter is required")
  logger.info(f"Delete request received for: {filename}")
  success = delete_professor_document(filename)
  if success:
    return JSONResponse({"success": True, "message": f"Deleted {filename}"})
  raise HTTPException(status_code=404, detail=f"Document not found or deletion failed: {filename}")

@app.post("/deldoc")
async def deldoc_post(request: DelDocRequest):
    return await deldoc(filename=request.filename)

# --- WebSocket endpoint ---
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
  await manager.connect(client_id, websocket)
  try:
    while True:
      data = await websocket.receive_text()
      logger.info(f"Received from {client_id}: {data}")

      # Stream LLM responses in a background thread so we don't block the event loop
      loop = asyncio.get_running_loop()

      def stream_and_send():
        try:
          # Notify client that the actual response is starting (separates from the processing ack)
          start_coro = manager.send_personal_message("<RESPONSE_START>", client_id)
          try:
            asyncio.run_coroutine_threadsafe(start_coro, loop).result(timeout=5)
          except Exception:
            # If start marker fails to send, continue streaming anyway
            logger.exception("Failed to send RESPONSE_START marker")

          for chunk in llm_manager.run(data):
            # Schedule send on the event loop thread-safely
            coro = manager.send_personal_message(chunk, client_id)
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            try:
                fut.result()
            except Exception:
              logger.exception("Failed to send LLM chunk to client")
          # Optionally send a final marker
          coro = manager.send_personal_message("<END_OF_STREAM>", client_id)
          asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception:
          logger.exception("LLM streaming failed")

      loop.run_in_executor(None, stream_and_send)

  except WebSocketDisconnect:
    manager.disconnect(client_id)
  except Exception as e:
    logger.exception("WebSocket error")
    manager.disconnect(client_id)

if __name__ == "__main__":
  uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

