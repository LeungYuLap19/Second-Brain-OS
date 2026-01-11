import sqlite3
import traceback
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from configs.settings_loader import settings
from langchain_ollama import ChatOllama

# Update this path if needed
DB_PATH = "C:/Users/ASUS/Documents/vscode/second brain OS/data/sqlite/accountant.db"

def get_connection():
  return sqlite3.connect(DB_PATH)

@tool
def add_transaction(amount: float, description: str, datetime: str = None) -> str:
  """Add a new transaction to the database.

  Use this tool whenever the user wants to record income or expense.
  - Positive amount = income
  - Negative amount = expense
  - datetime should be in ISO format (e.g., '2026-01-08 14:30:00'). If not provided, uses current time.

  Args:
    amount (float): Signed amount (e.g., -45.50 for expense, 2000.00 for income)
    description (str): Free-text description (e.g., "Starbucks coffee", "Salary January")
    datetime (str, optional): When the transaction occurred. Defaults to now.

  Returns:
    str: Confirmation message
  """
  # print(f"Adding Transaction: amount {amount} | description {description} | datetime {datetime}")
  conn = get_connection()
  cursor = conn.cursor()

  try:
    if datetime is None:
      cursor.execute("""
        INSERT INTO transactions (amount, description)
        VALUES (?, ?)
      """, (amount, description))
    else:
      cursor.execute("""
        INSERT INTO transactions (datetime, amount, description)
        VALUES (?, ?, ?)
      """, (datetime, amount, description))

    conn.commit()
    conn.close()
    return f"Successfully recorded: {amount:+.2f} — {description}"
  except Exception as e:
    conn.close()
    tb = traceback.format_exc()
    print("\n🔥 ADD_TRANSACTION ERROR 🔥\n", tb, "\n🔥 END 🔥\n")
    return f"Error recording transaction: {str(e)}"
    
@tool
def get_recent_transactions(limit: int = 10) -> str:
  """Retrieve the most recent transactions for review or confirmation.
  
  Args:
    limit (int): Number of recent transactions to return (default 10, max 50).
  
  Returns:
    str: Formatted list of recent transactions.
  """
  # print(f"Getting Transaction: limit {limit}")
  conn = get_connection()
  cursor = conn.cursor()
  safe_limit = min(max(1, limit), 50)
  
  cursor.execute("""
    SELECT datetime, amount, description 
    FROM transactions 
    ORDER BY datetime DESC 
    LIMIT ?
  """, (safe_limit,))
  
  rows = cursor.fetchall()
  conn.close()
  
  if not rows:
    return "No transactions recorded yet."
  
  lines = ["**Recent Transactions**"]
  for dt, amt, desc in rows:
    sign = "+" if amt > 0 else "-"
    lines.append(f"• {dt[:16]} | {sign}${abs(amt):.2f} | {desc}")
  
  return "\n".join(lines)

@tool
def search_transactions(keyword: str = None, start_date: str = None, end_date: str = None, limit: int = 20) -> str:
  """Search transactions by keyword in description or date range.
  
  Args:
    keyword (str, optional): Search term in specified requests (case-insensitive).
    start_date (str, optional): ISO date '2026-01-01'
    end_date (str, optional): ISO date '2026-02-01'
    limit (int): Max results (default 20).
  Returns:
    str: Matching transactions.
  """
  # print(f"Searching Transaction: keyword {keyword} | start_date {start_date} | end_date {end_date} | limit {limit}")
  conn = get_connection()
  cursor = conn.cursor()
  
  query = "SELECT datetime, amount, description FROM transactions WHERE 1=1"
  params = []
  
  if keyword:
    query += " AND description LIKE ?"
    params.append(f"%{keyword}%")
  if start_date:
    query += " AND datetime >= ?"
    params.append(f"{start_date} 00:00:00")
  if end_date:
    query += " AND datetime < ?"
    params.append(f"{end_date} 00:00:00")
  
  query += " ORDER BY datetime DESC LIMIT ?"
  params.append(limit)
  
  cursor.execute(query, params)
  rows = cursor.fetchall()
  conn.close()
  
  if not rows:
      return "No matching transactions found."
  
  lines = [f"**Search Results** ({len(rows)} found)"]
  for dt, amt, desc in rows:
    sign = "+" if amt > 0 else "-"
    lines.append(f"• {dt[:10]} | {sign}${abs(amt):.2f} | {desc}")

  return "\n".join(lines)

@tool
def delete_last_transaction() -> str:
  """Delete the most recent transaction (useful for immediate corrections).
  
  Returns:
    str: Confirmation or error.
  """
  # print(f"Deleting Last Transaction")
  conn = get_connection()
  cursor = conn.cursor()
  
  # Get the latest one first for confirmation
  cursor.execute("""
    SELECT datetime, amount, description 
    FROM transactions 
    ORDER BY datetime DESC LIMIT 1
  """)
  row = cursor.fetchone()
  
  if not row:
    conn.close()
    return "No transactions to delete."
  
  dt, amt, desc = row
  cursor.execute("DELETE FROM transactions WHERE datetime = ? AND amount = ? AND description = ?", (dt, amt, desc))
  
  if cursor.rowcount == 1:
    conn.commit()
    conn.close()
    return f"Deleted the last transaction: {amt:+.2f} — {desc} ({dt[:16]})"
  else:
    conn.close()
    return "Failed to delete — transaction may have been modified."
  
@tool
def summarize_month(year: int, month: int) -> str:
  """Get income, expenses, and net for a specific month.
  
  Args:
    year (int): e.g., 2026
    month (int): 1-12
  
  Returns:
    str: Summary with totals.
  """
  # print(f"Summarizing Month Transaction: year {year} | month {month}")
  conn = get_connection()
  cursor = conn.cursor()
  
  start = f"{year:04d}-{month:02d}-01 00:00:00"
  end = f"{year:04d}-{month:02d+1:02d}-01 00:00:00" if month < 12 else f"{year+1:04d}-01-01 00:00:00"
  
  cursor.execute("""
    SELECT 
      SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as income,
      SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as expenses
    FROM transactions 
    WHERE datetime >= ? AND datetime < ?
  """, (start, end))
  
  row = cursor.fetchone()
  conn.close()
  
  income = row[0] or 0.0
  expenses = row[1] or 0.0
  net = income - expenses
  
  month_name = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][month-1]
  
  return f"**{month_name} {year} Summary**\nIncome: +${income:.2f}\nExpenses: -${expenses:.2f}\nNet: {net:+.2f}"

# ------------------------------------------------------------------
# Keep execute_sql_write for advanced cases (editing/deleting by user request)
# But make it safer and clearer
# ------------------------------------------------------------------
@tool
def execute_sql_write(query: str) -> str:
  """Execute a single INSERT, UPDATE, or DELETE statement.

  Only use this for advanced modifications (e.g., correcting a past entry, bulk delete).
  Never use for SELECT — use SQL query tools instead.
  Never expose or use internal IDs in user-facing responses.

  Args:
      query (str): A single valid SQL write statement.

  Returns:
      str: Success or error message.
  """
  print(f"Executing SQL Write: query {query}")
  conn = get_connection()
  cursor = conn.cursor()

  try:
    cursor.execute(query)
    conn.commit()
    rows = cursor.rowcount
    conn.close()
    return f"Success: {rows} row(s) affected."
  except Exception as e:
    conn.close()
    tb = traceback.format_exc()
    print("\n🔥 EXECUTE_SQL_WRITE ERROR 🔥\n", tb, "\n🔥 END 🔥\n")
    return f"execute_sql_write error: {str(e)}"


# ------------------------------------------------------------------
# SQLDatabase toolkit setup (read-only operations)
# ------------------------------------------------------------------
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

agent_config = settings.get_agent_model_config("Accountant")
llm = ChatOllama(
  model=agent_config["model"],
  temperature=agent_config.get("temperature", 0.0),
  num_ctx=agent_config.get("num_ctx", 8192),
  num_predict=agent_config.get("num_predict", -1),
  base_url=settings.get_base_url(),
  format=agent_config.get("format", ""),
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Extract the standard SQL tools
sql_list_tables = next(t for t in toolkit.get_tools() if t.name == "sql_db_list_tables")
sql_get_schema = next(t for t in toolkit.get_tools() if t.name == "sql_db_schema")
sql_query = next(t for t in toolkit.get_tools() if t.name == "sql_db_query")
sql_query_checker = next(t for t in toolkit.get_tools() if t.name == "sql_db_query_checker")