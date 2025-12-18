from .tavily import tavily_search_api, tavily_extract_content

TOOL_REGISTRY = {
  "tavily_search_api": tavily_search_api,
  "tavily_extract_content": tavily_extract_content
}