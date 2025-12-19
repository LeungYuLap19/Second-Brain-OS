from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

VECTORSTORE_PATH = "data/vectordb"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

@tool
def search_documents(query: str, k: int = 5) -> str:
    """
    Search through uploaded documents (PDFs, Word, PPT, TXT, images) for relevant content.
    
    Args:
      query: The search question or topic
      k: Number of top relevant chunks to retrieve (default: 5)
    
    Returns:
      Formatted relevant excerpts with sources, or a clear message if nothing found.
    """
    # Load the vectorstore
    try:
      vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
      )
    except Exception as e:
      return f"Error loading document database: {str(e)}. Make sure documents have been ingested."

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    if not docs:
      return "No relevant information found in the uploaded documents."

    # Format results nicely
    results = []
    for i, doc in enumerate(docs, 1):
      source = doc.metadata.get('source', 'unknown file')
      # Make source relative and clean
      if source.startswith("data/"):
        source = source[len("data/"):]
      results.append(f"[Result {i}] From: {source}\n{doc.page_content}")

    return "\n\n".join(results)