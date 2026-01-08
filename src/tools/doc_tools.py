from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import traceback

PROFESSOR_VDB_PATH = "data/vectordb/professor"
MEMORY_VDB_PATH = "data/vectordb/memory"
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
        PROFESSOR_VDB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
      )
      
    except Exception as e:
      tb = traceback.format_exc()
      print("\n🔥 TASK FAILED TRACEBACK 🔥")
      print(tb)
      print("🔥 END TRACEBACK 🔥\n")
      return f"search_documents error: {str(e)}"

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

@tool
def search_memory(
    query: str,
    k: int = 5,
    mode: str = "auto",  # "auto", "full", "fine"
    agent_filter: str | None = None,
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

    k : int, default 5
        Number of memory chunks to retrieve. Use smaller k (2-4) when you're confident about the match.
        Use larger k (6-10) when searching broadly.

    mode : str, default "auto"
        Controls retrieval granularity:
        - "auto"  : Recommended. System chooses between full outputs or fine chunks based on query and relevance.
        - "full"  : Force retrieval of entire past agent outputs (ideal for continuing long reasoning chains or reusing structure).
        - "fine"  : Retrieve only small, precise snippets (useful for extracting a single fact, number, or short conclusion).

    agent_filter : str | None
        Filter by specific agent name (e.g., "researcher", "coder", "planner") if you only want outputs from that agent.

    step_filter : int | None
        Retrieve memory only from a specific step number (useful for exact recall of "step 7" or "iteration 12").

    Returns
    -------
    str
        Formatted string with relevant memories, marked READ-ONLY.
        Each entry shows index, agent, step, granularity, and full content.
        Use this to directly inform or structure your current response.

    Best Practices
    ---------------
    • Call this tool early — before planning or answering — whenever continuity matters.
    • When continuing a task, use mode="full" to recover complete previous context and maintain identical response style.
    • Quote or adapt relevant sections directly to ensure consistency.
    • If no relevant memory is returned, proceed with fresh reasoning.
    """
    print(
       f"{query}\n"
       f"{k}\n"
       f"{mode}\n"
       f"{agent_filter}\n"
       f"{step_filter}\n"
    )
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.load_local(
            MEMORY_VDB_PATH,
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
        if mode == "full":
            # Search only in full outputs
            results = vectorstore.similarity_search(
                query, k=k, filter={**filter_dict, "granularity": "full"}
            )
        elif mode == "fine":
            # Search only in fine chunks
            results = vectorstore.similarity_search(
                query, k=k, filter={**filter_dict, "granularity": "fine"}
            )
        else:  # "auto" - smart hybrid
            # First try to find matching full outputs
            full_results = vectorstore.similarity_search(
                query, k=3, filter={**filter_dict, "granularity": "full"}
            )

            if full_results and any(score < 0.25 for doc, score in vectorstore.similarity_search_with_score(query, k=3, filter={**filter_dict, "granularity": "full"})):
                # High confidence match on full output → return it
                results = full_results[:k]
            else:
                # Fall back to fine-grained chunks
                results = vectorstore.similarity_search(
                    query, k=k, filter={**filter_dict, "granularity": "fine"}
                )

        if not results:
            return ""

        chunks = []
        for i, doc in enumerate(results, 1):
            agent = doc.metadata.get("agent", "unknown")
            step = doc.metadata.get("step", "unknown")
            gran = doc.metadata.get("granularity", "?")
            chunks.append(
                f"[Memory {i}] (agent={agent}, step={step}, granularity={gran})\n"
                f"{doc.page_content}"
            )

        header = "### Relevant Past Memory (READ-ONLY)\n"
        if mode != "auto":
            header += f"(Retrieval mode: {mode})\n"

        return "\n\n" + header + "\n\n".join(chunks)

    except Exception as e:
        tb = traceback.format_exc()
        print("\n🔥 MEMORY RETRIEVAL FAILED 🔥")
        print(tb)
        print("🔥 END TRACEBACK 🔥\n")
        return f"search_memory error: {str(e)}"