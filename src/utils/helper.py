import re
from langchain_core.documents import Document
from langchain_community.document_loaders import (
  PyPDFLoader, Docx2txtLoader, TextLoader,
  UnstructuredPowerPointLoader, UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
import shutil

def clean_html_content(html_text: str) -> str:
  """
  Thoroughly clean HTML content to extract only readable text.
  Removes: HTML tags, CSS styles, JavaScript, comments, etc.
  """
  if not html_text:
      return ""
  
  # First decode HTML entities
  import html
  text = html.unescape(html_text)
  
  # Remove CSS content (style tags and inline style attributes)
  # Remove <style> tags and their content
  text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
  
  # Remove inline style attributes
  text = re.sub(r'style="[^"]*"', '', text, flags=re.IGNORECASE)
  text = re.sub(r"style='[^']*'", '', text, flags=re.IGNORECASE)
  
  # Remove script tags and their content
  text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
  
  # Remove HTML comments
  text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
  
  # Remove all other HTML tags but preserve text content
  text = re.sub(r'<[^>]+>', ' ', text)
  
  # Remove CSS block patterns (content between { and })
  text = re.sub(r'\{[^}]*\}', '', text)
  
  # Remove common CSS selectors that might appear in email body
  css_selectors = [
      r'\.\w+\s*{[^}]*}',
      r'#\w+\s*{[^}]*}',
      r'@media[^{]+\{[^}]*\}',
      r'@font-face[^{]+\{[^}]*\}',
      r'body\s*{[^}]*}',
      r'div\s*{[^}]*}',
      r'span\s*{[^}]*}',
      r'p\s*{[^}]*}',
      r'a\s*{[^}]*}',
      r'table\s*{[^}]*}',
      r'tr\s*{[^}]*}',
      r'td\s*{[^}]*}',
      r'th\s*{[^}]*}',
  ]
  for pattern in css_selectors:
      text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
  
  # Remove multiple spaces, newlines, tabs
  text = re.sub(r'\s+', ' ', text)
  
  # Remove common email artifacts
  text = re.sub(r'&nbsp;', ' ', text)
  text = re.sub(r'&zwnj;', '', text)
  text = re.sub(r'&amp;', '&', text)
  text = re.sub(r'&lt;', '<', text)
  text = re.sub(r'&gt;', '>', text)
  text = re.sub(r'&quot;', '"', text)
  text = re.sub(r'&#39;', "'", text)
  text = re.sub(r'&#\d+;', '', text)  # Remove numeric entities
  
  # Remove URLs (optional, but keeps text cleaner)
  # text = re.sub(r'https?://\S+', '', text)
  
  # Clean up whitespace again
  text = re.sub(r'\s+', ' ', text).strip()
  
  return text


# ----------------------------------------------------------


DATA_FOLDER = "data/input"
PROFESSOR_VDB_PATH = "data/vectordb/professor"
MEMORY_VDB_PATH = "data/vectordb/memory"
PROFESSOR_INDEX_FILE = os.path.join(PROFESSOR_VDB_PATH, "index.faiss")
MEMORY_INDEX_FILE = os.path.join(MEMORY_VDB_PATH, "index.faiss")

def get_loader(file_path: str):
  ext = os.path.splitext(file_path)[1].lower()
  if ext == ".pdf":
    return PyPDFLoader(file_path)
  elif ext in [".docx", ".doc"]:
    return Docx2txtLoader(file_path)
  elif ext in [".txt", ".md"]:
    return TextLoader(file_path, encoding="utf-8")
  elif ext in [".ppt", ".pptx"]:
    return UnstructuredPowerPointLoader(file_path)
  else:
    return UnstructuredFileLoader(file_path) # handles images, etc. with OCR if Tesseract installed
  
def ingest_documents_generic(
  documents: list[Document],
  vectorstore_path: str,
  index_file: str,
):
  if not documents:
    print("No documents to ingest.")
    return

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
  )
  splits = text_splitter.split_documents(documents)

  embeddings = OllamaEmbeddings(model="nomic-embed-text")

  if os.path.exists(index_file):
    print(f"Loading existing vectorstore at {vectorstore_path}...")
    vectorstore = FAISS.load_local(
      vectorstore_path,
      embeddings,
      allow_dangerous_deserialization=True
    )
    vectorstore.add_documents(splits)
  else:
    print(f"Creating new vectorstore at {vectorstore_path}...")
    vectorstore = FAISS.from_documents(splits, embeddings)

  vectorstore.save_local(vectorstore_path)
  print("Ingestion complete.")

def ingest_professor_documents():
  docs = []

  for filename in os.listdir(DATA_FOLDER):
    file_path = os.path.join(DATA_FOLDER, filename)
    if not os.path.isfile(file_path):
      continue

    loader = get_loader(file_path)
    docs.extend(loader.load())

  ingest_documents_generic(
    documents=docs,
    vectorstore_path=PROFESSOR_VDB_PATH,
    index_file=PROFESSOR_INDEX_FILE
  )

from datetime import datetime, timezone

def ingest_memory_texts(
    texts: list[str],
    metadatas: list[dict] | None = None,
):
    documents = []

    for i, text in enumerate(texts):
        base_metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
        base_metadata.update({
            "type": "memory",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_index": i,  # links small chunks back to their full output
        })

        # 1. Store the FULL output as one large document
        documents.append(
            Document(
                page_content=text,
                metadata={**base_metadata, "granularity": "full", "agent": base_metadata.get("agent", "unknown"), "step": base_metadata.get("step", "unknown")}
            )
        )

        # 2. Split into small chunks for precise retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,      # smaller for precise recall
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "? ", "! ", " "]
        )
        splits = text_splitter.split_text(text)

        for j, split in enumerate(splits):
            documents.append(
                Document(
                    page_content=split,
                    metadata={
                        **base_metadata,
                        "granularity": "fine",
                        "chunk_index": j,
                        "agent": base_metadata.get("agent", "unknown"),
                        "step": base_metadata.get("step", "unknown")
                    }
                )
            )

    ingest_documents_generic(
        documents=documents,
        vectorstore_path=MEMORY_VDB_PATH,
        index_file=MEMORY_INDEX_FILE
    )

def clear_memory_vdb():
  """
  Completely resets the memory vector database.

  Intended for:
  - testing
  - debugging
  - development resets

  WARNING:
  This permanently deletes ALL stored memory embeddings.
  """
  if not os.path.exists(MEMORY_VDB_PATH):
    print("Memory VDB does not exist. Nothing to clear.")
    return

  shutil.rmtree(MEMORY_VDB_PATH)
  os.makedirs(MEMORY_VDB_PATH, exist_ok=True)

  print("Memory VDB cleared.")
