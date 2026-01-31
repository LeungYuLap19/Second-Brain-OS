import os
import shutil 
import html 
import re
from typing import List, Optional
from langchain_ollama import OllamaEmbeddings
from configs.settings_loader import settings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from configs.paths import RAG_VECTOR_DB_DIR, RAG_INDEX_FILE, INPUT_DATA_DIR

def clean_html_content(html_text: str) -> str:
    """
    Thoroughly clean HTML content to extract only readable text.
    
    Removes:
    - HTML tags
    - CSS styles (both <style> tags and inline styles)
    - JavaScript (<script> tags)
    - HTML comments
    - CSS block patterns and selectors
    - Common HTML entities (&nbsp;, &amp;, etc.)
    
    Args:
        html_text (str): Raw HTML content to clean
        
    Returns:
        str: Clean, readable text with all HTML/CSS/JS removed
        
    Example:
        >>> html = "<p>Hello <strong>world</strong></p>"
        >>> clean_html_content(html)
        'Hello world'
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


# ---------- VECTORDB HELPERS ---------- #

# Embedding model configuration for document retrieval
embedding_config = settings.get_embedding_model()
EMBEDDINGS = OllamaEmbeddings(
    model=embedding_config['model'],
    base_url=embedding_config.get('base_url')
)

def get_loader(file_path: str):
    """
    Get the appropriate document loader based on file extension.
    
    Supports:
    - PDF (.pdf) → PyPDFLoader
    - Word documents (.docx, .doc) → Docx2txtLoader
    - Text files (.txt, .md) → TextLoader
    - PowerPoint (.ppt, .pptx) → UnstructuredPowerPointLoader
    - Other files → UnstructuredFileLoader (fallback)
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        DocumentLoader: Appropriate loader instance for the file type
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
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
    """
    Ingest documents into a FAISS vector store.
    
    Handles both:
    1. Creating a new vector store if none exists
    2. Adding to an existing vector store
    
    Args:
        documents (List[Document]): List of Document objects to ingest
        vectorstore_path (str): Directory path where vector store is saved
        index_file (str): Path to the FAISS index file
        
    Returns:
        None
        
    Note:
        Uses `allow_dangerous_deserialization=True` to load existing FAISS stores.
        This is safe for trusted sources but be cautious with untrusted files.
    """
    if not documents:
        return

    if os.path.exists(index_file):
        # Load existing vector store and add new documents
        vectorstore = FAISS.load_local(
            vectorstore_path,
            EMBEDDINGS,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(documents)
    else:
        # Create new vector store
        vectorstore = FAISS.from_documents(documents, EMBEDDINGS)

def ingest_professor_document(filename: str) -> bool:
    """
    Process and ingest a document into the RAG (Retrieval-Augmented Generation) system.
    
    Workflow:
    1. Load document from INPUT_DATA_DIR
    2. Split into chunks with overlap
    3. Add metadata (filename, chunk index)
    4. Ingest into vector store
    
    Args:
        filename (str): Name of the file in INPUT_DATA_DIR to ingest
        
    Returns:
        bool: True if ingestion was successful, False otherwise
        
    Example:
        >>> ingest_professor_document("research_paper.pdf")
        True
    """
    file_path = os.path.join(INPUT_DATA_DIR, filename)
    if not os.path.isfile(file_path):
        return False

    loader = get_loader(file_path)
    raw_doc = loader.load()

    if not raw_doc:
        return False

    # Split document into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # 1000 characters per chunk
        chunk_overlap=200     # 200 character overlap between chunks
    )
    splits = text_splitter.split_documents(raw_doc)

    # Add metadata to each chunk
    documents = []
    for i, split in enumerate(splits):
        documents.append(
            Document(
                page_content=split.page_content,
                metadata={
                    "filename": filename,
                    "chunk_index": i  # Track order of chunks
                }
            )
        )

    # Ingest into vector store
    ingest_documents_generic(
        documents=documents,
        vectorstore_path=RAG_VECTOR_DB_DIR,
        index_file=RAG_INDEX_FILE
    )
    
    return True

def delete_professor_document(filename: str) -> bool:
    """
    Remove a document and all its chunks from the RAG vector store.
    
    Workflow:
    1. Load existing vector store
    2. Filter out all chunks with matching filename
    3. Either:
       - Delete entire store if no documents remain
       - Save filtered store if documents remain
    4. Delete original file from INPUT_DATA_DIR
    
    Args:
        filename (str): Name of the file to delete
        
    Returns:
        bool: True if deletion was successful, False if file wasn't found
        
    Example:
        >>> delete_professor_document("old_contract.pdf")
        True  # Document removed
    """
    if not os.path.exists(RAG_VECTOR_DB_DIR):
        return False

    # Load existing vector store
    vectorstore = FAISS.load_local(
        RAG_VECTOR_DB_DIR,
        EMBEDDINGS,
        allow_dangerous_deserialization=True
    )

    # Get all documents from the store
    docs = list(vectorstore.docstore._dict.values())

    if not docs:
        return False

    # Filter out documents with matching filename
    remaining_docs = [
        doc for doc in docs
        if doc.metadata.get("filename") != filename
    ]

    # If nothing removed, file wasn't indexed
    if len(remaining_docs) == len(docs):
        return False

    # If everything removed, wipe the entire database
    if not remaining_docs:
        shutil.rmtree(RAG_VECTOR_DB_DIR, ignore_errors=True)
    else:
        # Create new vector store with remaining documents
        new_vs = FAISS.from_documents(remaining_docs, EMBEDDINGS)
        new_vs.save_local(RAG_VECTOR_DB_DIR)

    # Delete original file from input directory
    file_path = os.path.join(INPUT_DATA_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return True