from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from typing import Literal
import traceback
import os

PROFESSOR_VDB_PATH = "data/vectordb/professor"
MEMORY_VDB_FULL_PATH = "data/vectordb/memory_full"
MEMORY_VDB_CHUNKS_PATH  = "data/vectordb/memory_chunks"

PROFESSOR_INDEX_FILE = os.path.join(PROFESSOR_VDB_PATH, "index.faiss")
MEMORY_FULL_INDEX_FILE = os.path.join(MEMORY_VDB_FULL_PATH, "index.faiss")
MEMORY_CHUNKS_INDEX_FILE = os.path.join(MEMORY_VDB_CHUNKS_PATH, "index.faiss")

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
      if not os.path.exists(PROFESSOR_INDEX_FILE):
        return ""

      vectorstore = FAISS.load_local(
        PROFESSOR_VDB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
      )

      retriever = vectorstore.as_retriever(search_kwargs={"k": k})
      docs = retriever.invoke(query)

      if not docs:
        return "No relevant information found in the uploaded documents."

      # Format results nicely
      results = []
      for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'unknown file')
        if source.startswith("data/"):
          source = source[len("data/"):]
        results.append(f"[Result {i}] From: {source}\n{doc.page_content}")

      return "\n\n".join(results)
      
    except Exception as e:
      tb = traceback.format_exc()
      print("\n🔥 TASK FAILED TRACEBACK 🔥")
      print(tb)
      print("🔥 END TRACEBACK 🔥\n")
      return f"search_documents error: {str(e)}"

@tool
def search_memory(
  query: str,
  mode: Literal["full", "fine"],
  k: int = 5,
  agent_filter: Literal["Professor", "Researcher", "Communicator", "Secretary", "Accountant", "Responder"] | None = None,
  step_filter: int | None = None
) -> str:
  """
  Retrieve the most relevant past agent outputs (or precise chunks) from long-term memory.

  This vector database stores the **full, complete outputs** of previous agent steps — including all reasoning, plans, code, results, decisions, and conclusions.  
  These full outputs contain far more detail than the short summaries shown in State Memory.

  You should almost always call search_memory first when:
  • The current task continues, builds on, or references previous work
  • You need to recall your exact previous reasoning, code, findings, or decisions
  • You want to maintain consistent style, structure, or formatting across steps
  • The user asks about "before", "earlier", "last time", "previously", or refers to past results
  • You see familiar numbers, filenames, conclusions, errors, or plans

  The State Memory you receive only shows brief summaries of past turns.  
  To access the complete original agent response (with full context, intermediate thoughts, and exact output), use this tool.

  Parameters
  ----------
  query : str
    A natural-language description of what you're looking for.
    Good queries include key phrases, conclusions, numbers, code snippets, filenames, or task descriptions from past outputs.
    Example: "full reasoning and final plan for the data preprocessing pipeline"
              "code I wrote to fix the OAuth token refresh bug"
              "conclusion about model accuracy on the test set"

  mode : str
    Controls retrieval granularity:
    - "full" : Retrieval of entire past agent outputs (ideal for continuing long reasoning chains or reusing structure, Especially plans, itineraries).
    - "fine" : Retrieve only small, precise snippets (useful for extracting a single fact, number, or short conclusion).

  k : int, default 5
    Number of memory chunks to retrieve. Use smaller k (2-4) when you're confident about the match.
    Use larger k (6-10) when searching broadly.

  agent_filter : str | None
    Filter by specific agent name ("Professor", "Researcher", "Communicator", "Secretary", "Accountant", "Responder") if you only want outputs from that agent.

  step_filter : int | None
    Retrieve memory only from a specific step number (useful for exact recall of "step 7" or "iteration 12").

  Returns
  -------
  str
    Formatted string with relevant memories, marked READ-ONLY.
    Each entry shows index, agent, step, and full content.
    Use this to directly inform or structure your current response.

  Best Practices
  ---------------
  • Call this tool early — before planning or answering — whenever continuity matters.
  • When continuing a task, use mode="full" to recover complete previous context and maintain identical response style.
  • Quote or adapt relevant sections directly to ensure consistency.
  • If no relevant memory is returned, proceed with fresh reasoning.
  """
  # print(
  #   f"query: {query}\n"
  #   f"mode: {mode}\n"
  #   f"agent_filter: {agent_filter}\n"
  #   f"step_filter: {step_filter}\n"
  #   f"k: {k}\n"
  # )

  try:
    index_file = MEMORY_FULL_INDEX_FILE if mode == "full" else MEMORY_CHUNKS_INDEX_FILE
    if not os.path.exists(index_file):
      return ""
    
    path = MEMORY_VDB_FULL_PATH if mode == "full" else MEMORY_VDB_CHUNKS_PATH
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local(
      path,
      embeddings,
      allow_dangerous_deserialization=True
    )

    # Build filter
    filter_dict = {}
    if agent_filter:
      filter_dict["agent"] = agent_filter
    if step_filter is not None:
      filter_dict["step"] = step_filter

    # Strategy based on mode
    results = vectorstore.similarity_search(
      query,
      k=k,
      filter=filter_dict if filter_dict else None
    )

    if not results:
      return ""

    chunks = []
    for i, doc in enumerate(results, 1):
      agent = doc.metadata.get("agent", "unknown")
      step = doc.metadata.get("step", "unknown")
      chunk_index = doc.metadata.get("chunk_index", "N/A")
      chunks.append(
        f"[Memory {i}] (agent={agent}, step={step}, chunk_index={chunk_index})\n"
        f"{doc.page_content}"
      )

    header = "### Relevant Past Memory (READ-ONLY)\n"
    header += f"(Retrieval mode: {mode})\n"

    return "\n\n" + header + "\n\n".join(chunks)

  except Exception as e:
    tb = traceback.format_exc()
    print("\n🔥 MEMORY RETRIEVAL FAILED 🔥")
    print(tb)
    print("🔥 END TRACEBACK 🔥\n")
    return f"search_memory error: {str(e)}"