from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader,
    UnstructuredPowerPointLoader, UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

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