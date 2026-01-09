from .tavily import tavily_search_api, tavily_extract_content
from .doc_tools import search_documents, search_memory
from .gmail import get_emails, gmail_send_message
from .calendar import search_calendar_events, create_calendar_event, update_calendar_event, delete_calendar_event
from .sqlite import add_transaction, get_recent_transactions, search_transactions, delete_last_transaction, summarize_month, execute_sql_write, sql_list_tables, sql_get_schema, sql_query, sql_query_checker

TOOL_REGISTRY = {
  "tavily_search_api": tavily_search_api,
  "tavily_extract_content": tavily_extract_content,

  "search_documents": search_documents,
  "search_memory": search_memory,

  "get_emails": get_emails,
  "gmail_send_message": gmail_send_message,

  "search_calendar_events": search_calendar_events,
  "create_calendar_event": create_calendar_event,
  "update_calendar_event": update_calendar_event,
  "delete_calendar_event": delete_calendar_event,

  "add_transaction": add_transaction,
  "get_recent_transactions": get_recent_transactions,
  "search_transactions": search_transactions,
  "delete_last_transaction": delete_last_transaction,
  "summarize_month": summarize_month,
  "execute_sql_write": execute_sql_write,
  "sql_list_tables": sql_list_tables,
  "sql_get_schema": sql_get_schema,
  "sql_query": sql_query,
  "sql_query_checker": sql_query_checker,
}