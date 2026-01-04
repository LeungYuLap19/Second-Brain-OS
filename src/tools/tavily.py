from tavily import TavilyClient
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
import traceback

load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def tavily_search_api(query: str, max_results: int = 3) -> str:
  """
  Perform a web search using the Tavily Search API and return structured results.

  This tool is useful when you need up-to-date information from the web, 
  such as current events, facts, statistics, or answers to factual questions 
  that may not be available in the model's training data.

  Args:
    query (str): The search query to send to Tavily. Should be clear and specific.
    max_results (int, optional): Maximum number of search results to return. 
                                Defaults to 3 to keep responses concise.

  Returns:
    str: A JSON-serializable object (list of dicts or dict) containing the search results.
          Typically a list where each result includes:
          - "title": Page title
          - "url": Source URL
          - "content": A short snippet or summary of the page
          If an error occurs, returns a dict with an "error" key.

  Note:
    The tool uses Tavily's "basic" search depth for faster results.
    In case of API errors, the exception traceback is printed to console 
    for debugging, and an error dictionary is returned.
  """
  try:
    response = tavily_client.search(
      query=query,
      max_results=max_results,
      search_depth="basic"
    )
    # "results" usually contains an array of search results with title, url, and snippet/content
    if "results" in response:
      return response["results"]
    return response
  except Exception as e:
    tb = traceback.format_exc()
    print("\n🔥 TASK FAILED TRACEBACK 🔥")
    print(tb)
    print("🔥 END TRACEBACK 🔥\n")
    return f"tavily_search_api error: {e}"
  
@tool
def tavily_extract_content(urls: list[str], include_images: bool = False) -> str:
  """
  Extract raw text content from one or more webpages using Tavily's Extract API.

  This tool is helpful when you have specific URLs from a prior search 
  (or elsewhere) and need the full cleaned text content rather than just snippets.
  It removes boilerplate (navigation, ads, etc.) and returns readable article text.

  Args:
    urls (list[str]): A list of URLs to extract content from. 
                      Must be valid, publicly accessible web pages.
    include_images (bool, optional): Whether to include image metadata 
                                      (URLs, alt text) in the extraction. 
                                      Defaults to False for cleaner text output.

  Returns:
    str: A JSON-serializable object containing the extraction results.
        Typically a list of dicts with keys like "url" and "raw_content".
        If an error occurs, returns a dict with an "error" key.

  Note:
    Uses Tavily's "advanced" extraction depth for high-quality cleaned content.
    Errors are caught, logged with full traceback to console, 
    and returned as an error dictionary.
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
    tb = traceback.format_exc()
    print("\n🔥 TASK FAILED TRACEBACK 🔥")
    print(tb)
    print("🔥 END TRACEBACK 🔥\n")
    return f"tavily_extract_content error: {e}"