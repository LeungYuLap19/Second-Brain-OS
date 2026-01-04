import sqlite3
import traceback
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from configs.settings_loader import settings
from langchain_ollama import ChatOllama

DB_PATH = "C:/Users/ASUS/Documents/vscode/second brain OS/data/sqlite/accountant.db"

def get_connection():
  return sqlite3.connect(DB_PATH)

@tool
def create_category(name: str) -> str:
  """Create a new category in the accountant database.

  This tool should be used whenever the user explicitly requests to add a new spending or income category
  (e.g., "Add a category called Travel", "Create category Gym", "New category for Subscriptions").

  The tool performs a case-insensitive check to prevent duplicate categories and returns a clear message
  if the category already exists.

  Args:
    name (str): The name of the new category. Case-insensitive uniqueness is enforced.

  Returns:
    str: A confirmation message if the category was created, or a message indicating it already exists.

  Examples:
    >>> create_category("Food")
    "Successfully added category: Food"

    >>> create_category("food")  # if "Food" already exists
    "Category 'food' already exists."
  """
  connection = get_connection()
  cursor = connection.cursor()

  # Case-insensitive duplicate check
  cursor.execute("SELECT 1 FROM categories WHERE LOWER(name) = LOWER(?)", (name,))
  if cursor.fetchone():
    connection.close()
    return f"Category '{name}' already exists."

  cursor.execute("INSERT INTO categories (name) VALUES (?)", (name,))
  connection.commit()
  connection.close()
  return f"Successfully added category: {name}"


@tool
def execute_sql_write(query: str) -> str:
  """Execute a single INSERT, UPDATE, or DELETE statement on the accountant database.

  Use this tool ONLY for data modifications (adding, editing, or deleting transactions, categories, etc.).
  NEVER use it for SELECT queries — use the standard SQL query tools for reading data.

  Critical guidelines (must be followed for correct behaviour):
  - Amounts must be signed: negative for expenses (e.g., -45.00), positive for income (e.g., 1000.00).
  - Date format must be DD-MM-YYYY (e.g., '03-01-2026').
  - Always look up category_id using a SELECT query first if inserting/updating a transaction.
  - Only one statement per call (no multi-statement or scripting).
  - Prefer this tool for transaction modifications when precise control is needed.

  Args:
    query (str): A single valid SQL INSERT, UPDATE, or DELETE statement.

  Returns:
    str: Success message with number of rows affected, or an error message if the query fails.

  Examples:
    >>> execute_sql_write(
    ...     "INSERT INTO transactions (date, amount, type, description, category_id, notes) "
    ...     "VALUES ('03-01-2026', -45.00, 'expense', 'Coffee', 1, NULL)"
    ... )
    "Success: 1 row(s) affected."

    >>> execute_sql_write(
    ...     "UPDATE transactions SET amount = -60.00 WHERE id = 42"
    ... )
    "Success: 1 row(s) affected."
  """
  connection = get_connection()
  cursor = connection.cursor()

  try:
    cursor.execute(query)
    connection.commit()
    rows_affected = cursor.rowcount
    connection.close()
    return f"Success: {rows_affected} row(s) affected."
  except Exception as e:
    tb = traceback.format_exc()
    print("\n🔥 TASK FAILED TRACEBACK 🔥")
    print(tb)
    print("🔥 END TRACEBACK 🔥\n")
    return f"execute_sql_write error: {str(e)}"
  
# SQLDatabase setup for read-only operations
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
agent_config = settings.get_agent_model_config("Accountant")
llm = ChatOllama(
  model=agent_config["model"],
  temperature=agent_config.get("temperature", 0.7),
  num_ctx=agent_config.get("num_ctx", 4096),
  num_predict=agent_config.get("num_predict"),
  base_url=settings.get_base_url(),
  format=agent_config.get("format", ""),
)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

list_tables_tool = next(t for t in toolkit.get_tools() if t.name == "sql_db_list_tables")
schema_tool = next(t for t in toolkit.get_tools() if t.name == "sql_db_schema")
query_tool = next(t for t in toolkit.get_tools() if t.name == "sql_db_query")
query_checker_tool = next(t for t in toolkit.get_tools() if t.name == "sql_db_query_checker")

sql_list_tables = list_tables_tool
sql_get_schema = schema_tool
sql_query = query_tool
sql_query_checker = query_checker_tool