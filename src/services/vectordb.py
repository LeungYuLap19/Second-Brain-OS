import os
import shutil
from typing import List

from langchain_ollama import OllamaEmbeddings
from configs.settings_loader import settings
from langchain_core.documents import Document
from langchain_community.document_loaders import (
	PyPDFLoader,
	Docx2txtLoader,
	TextLoader,
	UnstructuredPowerPointLoader,
	UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from configs.paths import RAG_VECTOR_DB_DIR, RAG_INDEX_FILE, INPUT_DATA_DIR


# Embedding model configuration for document retrieval
_embedding_config = settings.get_embedding_model()
EMBEDDINGS = OllamaEmbeddings(
	model=_embedding_config["model"],
	base_url=_embedding_config.get("base_url"),
)


def get_loader(file_path: str):
	"""Return appropriate document loader based on file extension."""
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
		return UnstructuredFileLoader(file_path)


def ingest_documents_generic(
	documents: List[Document],
	vectorstore_path: str,
	index_file: str,
) -> None:
	"""Ingest documents into a FAISS vector store.

	If an existing index exists, load it, add documents and save. Otherwise,
	create a new FAISS index and save it to disk.
	"""
	if not documents:
		return

	if os.path.exists(index_file):
		vectorstore = FAISS.load_local(
			vectorstore_path,
			EMBEDDINGS,
			allow_dangerous_deserialization=True,
		)
		vectorstore.add_documents(documents)
		vectorstore.save_local(vectorstore_path)
	else:
		vectorstore = FAISS.from_documents(documents, EMBEDDINGS)
		vectorstore.save_local(vectorstore_path)


def ingest_professor_document(filename: str) -> bool:
	"""Load a file from INPUT_DATA_DIR, split into chunks and ingest to RAG DB."""
	file_path = os.path.join(INPUT_DATA_DIR, filename)
	if not os.path.isfile(file_path):
		return False

	loader = get_loader(file_path)
	raw_doc = loader.load()
	if not raw_doc:
		return False

	splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
	splits = splitter.split_documents(raw_doc)

	documents = []
	for i, split in enumerate(splits):
		documents.append(
			Document(
				page_content=split.page_content,
				metadata={"filename": filename, "chunk_index": i},
			)
		)

	ingest_documents_generic(
		documents=documents,
		vectorstore_path=RAG_VECTOR_DB_DIR,
		index_file=RAG_INDEX_FILE,
	)

	return True

def delete_professor_document(filename: str) -> bool:
	"""Remove a document and its chunks from the RAG vector store and delete the source file."""
	if not os.path.exists(RAG_VECTOR_DB_DIR):
		return False

	vectorstore = FAISS.load_local(
		RAG_VECTOR_DB_DIR,
		EMBEDDINGS,
		allow_dangerous_deserialization=True,
	)

	docs = list(vectorstore.docstore._dict.values())
	if not docs:
		return False

	remaining_docs = [doc for doc in docs if doc.metadata.get("filename") != filename]

	if len(remaining_docs) == len(docs):
		return False

	if not remaining_docs:
		shutil.rmtree(RAG_VECTOR_DB_DIR, ignore_errors=True)
	else:
		new_vs = FAISS.from_documents(remaining_docs, EMBEDDINGS)
		new_vs.save_local(RAG_VECTOR_DB_DIR)

	file_path = os.path.join(INPUT_DATA_DIR, filename)
	if os.path.exists(file_path):
		os.remove(file_path)

	return True
