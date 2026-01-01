from .tavily import tavily_search_api, tavily_extract_content
from .doc_tools import search_documents
from .gmail import get_emails, gmail_send_message

TOOL_REGISTRY = {
  "tavily_search_api": tavily_search_api,
  "tavily_extract_content": tavily_extract_content,
  "search_documents": search_documents,
  "get_emails": get_emails,
  "gmail_send_message": gmail_send_message
}