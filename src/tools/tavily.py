from tavily import TavilyClient
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def tavily_search_api(query: str, max_results: int = 5, search_depth: str = "advanced") -> str:
  """
  Perform a web search using Tavily and return structured results.
  Returns a string or JSON-serializable dict with title + snippet.
  """
  try:
    response = tavily_client.search(
      query=query,
      max_results=max_results,
      search_depth=search_depth
    )
    # "results" usually contains an array of search results with title, url, and snippet/content
    if "results" in response:
      return response["results"]
    return response
  except Exception as e:
    return {"error": str(e)}
  
@tool
def tavily_extract_content(urls: list[str], include_images: bool = False) -> str:
  """
  Use Tavily Extract API to fetch content from the given URLs.
  Returns extracted text content.
  """
  try:
    response = tavily_client.extract(
      urls=urls,
      include_images=include_images,
      extract_depth="advanced"
    )
    if "results" in response:
      return response["results"]
    return response
  except Exception as e:
    return {"error": str(e)}