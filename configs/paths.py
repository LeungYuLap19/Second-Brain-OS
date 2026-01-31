import os
import sys

BASE_DIR = "/Users/jimmyleung/Documents/vscode/Second-Brain-OS-Server/"

CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
SYSTEM_PROMPT_FILE = os.path.join(CONFIGS_DIR, "system_prompt.txt")
LLM_MODEL_FILE = os.path.join(CONFIGS_DIR, "llm_model.yaml")

DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DATA_DIR = os.path.join(DATA_DIR, "input")
SQLITE_DIR = os.path.join(DATA_DIR, "sqlite")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vectordb")

RAG_VECTOR_DB_DIR = os.path.join(VECTOR_DB_DIR, "rag")
RAG_INDEX_FILE = os.path.join(RAG_VECTOR_DB_DIR, "index.faiss")

SQL_DIR = os.path.join(BASE_DIR, "sql")

ENV_FILE = os.path.join(BASE_DIR, ".env")
CREDENTIALS_FILE = os.path.join(BASE_DIR, "credentials.json")
TOKEN_FILE = os.path.join(BASE_DIR, "token.json")