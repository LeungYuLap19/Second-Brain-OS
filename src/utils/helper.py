import re
from langchain_community.document_loaders import (
  PyPDFLoader, Docx2txtLoader, TextLoader,
  UnstructuredPowerPointLoader, UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

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
VECTORSTORE_PATH = "data/vectordb"
INDEX_FILE = os.path.join(VECTORSTORE_PATH, "index.faiss")

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
  
def ingest_documents():
  docs = []
  for filename in os.listdir(DATA_FOLDER):
      file_path = os.path.join(DATA_FOLDER, filename)
      loader = get_loader(file_path)
      docs.extend(loader.load())

  # splitter and embedding model
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  splits = text_splitter.split_documents(docs)
  embeddings = OllamaEmbeddings(model="nomic-embed-text")

  # Only load if the actual index file exists
  if os.path.exists(INDEX_FILE):
    print("Loading existing vectorstore and adding new documents...")
    vectorstore = FAISS.load_local(
      VECTORSTORE_PATH,
      embeddings,
      allow_dangerous_deserialization=True
    )
    vectorstore.add_documents(splits)
  else:
    print("Creating new vectorstore from documents...")
    vectorstore = FAISS.from_documents(splits, embeddings)
  
  vectorstore.save_local(VECTORSTORE_PATH)
  print("Ingestion complete!")