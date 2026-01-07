from .tavily import tavily_search_api, tavily_extract_content
from .doc_tools import search_documents, search_memory
from .gmail import get_emails, gmail_send_message
from .sqlite import create_category, execute_sql_write, sql_list_tables, sql_get_schema, sql_query, sql_query_checker

TOOL_REGISTRY = {
  "tavily_search_api": tavily_search_api,
  "tavily_extract_content": tavily_extract_content,
  "search_documents": search_documents,
  "get_emails": get_emails,
  "gmail_send_message": gmail_send_message,
  "create_category": create_category,
  "execute_sql_write": execute_sql_write,
  "sql_list_tables": sql_list_tables,
  "sql_get_schema": sql_get_schema,
  "sql_query": sql_query,
  "sql_query_checker": sql_query_checker,
  "search_memory": search_memory
}