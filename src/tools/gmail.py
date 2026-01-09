import os.path
import base64
import re
import traceback
from email import policy
from email.parser import BytesParser
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain.tools import tool
from ..utils.helper import clean_html_content
from email.message import EmailMessage
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = [
  "https://www.googleapis.com/auth/gmail.readonly",
  "https://www.googleapis.com/auth/gmail.send"
]

def get_creds(scope: list[str]):
  creds = None
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", scope)
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
        "credentials.json", scope
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  return creds

def get_email(service, id: str):
  try:
    result = (
      service.users().messages().get(
        userId="me", 
        id=id,
        format="raw"
      ).execute()
    )
    raw_message = result.get("raw", {})

    if not raw_message:
      print("No message found.")
      return
    
    msg_bytes = base64.urlsafe_b64decode(raw_message)
    msg = BytesParser(policy=policy.default).parsebytes(msg_bytes)
    
    subject = msg["subject"]
    from_ = msg["from"]
    date = msg["date"]
    
    body = ""
    if msg.is_multipart():
      for part in msg.iter_parts():
        if part.get_content_type() == "text/plain":
          body = part.get_content()
          break
        elif part.get_content_type() == "text/html" and not body:
          html_content = part.get_content()
          body = re.sub(r'<[^>]+>', '', html_content)
    else:
        body = msg.get_content()
    
    return {
      "subject": subject,
      "from": from_,
      "date": date,
      "body": clean_html_content(body)[:1000] if body else ""
    }
  
  except HttpError as e:
    tb = traceback.format_exc()
    print("\n🔥 TASK FAILED TRACEBACK 🔥")
    print(tb)
    print("🔥 END TRACEBACK 🔥\n")
    return f"get_email error: {str(e)}"

@tool
def get_emails(
  query: str = None, 
  labelIds: str | list[str] = "INBOX",
  maxResults: int = 10,
  full_messages_str: bool = True,
  includeSpamTrash: bool = False,
  from_address: str = None,  
  subject_contains: str = None,
  after_date: str = None, 
  has_attachment: bool = None,
) -> str:
  """
  Fetch emails from Gmail with flexible filtering.
  
  Main filtering is done via the 'query' parameter (Gmail search syntax).
  
  Quick examples of valid query strings:
    "from:john.doe@example.com"
    "subject:invoice after:2025/01/01"
    "filename:pdf larger:5M"
    "is:unread label:Work"
    "-in:trash -in:sent"           # exclude trash & sent
    "category:updates older_than:1y"
  
  Convenience parameters (from_, subject_contains, etc.) will be automatically
  added to the query if provided.
  
  Args:
    query: Gmail search query string (most powerful option)
    labelIds: label name or list of labels (e.g. "INBOX", ["INBOX", "IMPORTANT"])
    maxResults: maximum number of messages to return (default 10)
    full_messages_str: whether to return concatenated full text content
    includeSpamTrash: include messages from spam/trash (default False)
    from_address: convenience filter - from specific sender
    subject_contains: convenience filter - subject contains words
    after_date: convenience filter - messages after this date (yyyy/mm/dd)
    has_attachment: convenience filter - messages with attachments
  
  Returns:
    If full_messages_str=True  → concatenated string of email contents
    Else                       → list of raw message metadata dicts
  """
  creds = get_creds(SCOPES)
  try:
    service = build("gmail", "v1", credentials=creds)

    # Build query parts
    query_parts = []

    if query:
      query_parts.append(query)

    # Convenience filters
    if from_address:
      query_parts.append(f"from:{from_address}")
    if subject_contains:
      query_parts.append(f"subject:{subject_contains}")
    if after_date:
      query_parts.append(f"after:{after_date}")
    if has_attachment is not None:
      query_parts.append("has:attachment" if has_attachment else "-has:attachment")

    # Combine all query parts
    final_query = " ".join(query_parts) if query_parts else None

    # Handle labelIds - can be string or list
    labels = [labelIds] if isinstance(labelIds, str) else labelIds

    # List messages
    results = service.users().messages().list(
      userId="me",
      q=final_query,
      labelIds=labels,
      maxResults=maxResults,
      includeSpamTrash=includeSpamTrash
    ).execute()

    messages = results.get("messages", [])

    if not messages:
      return "No messages found matching the criteria."

    if not full_messages_str:
      return messages

    # Full content mode
    email_contents = []
    for msg in messages:
      email_data = get_email(service, msg["id"])
      if email_data:
        email_contents.append(
          f"Message ID: {msg['id']}\n"
          f"Subject: {email_data.get('subject', 'No Subject')}\n"
          f"From: {email_data.get('from', 'Unknown')}\n"
          f"Date: {email_data.get('date', 'Unknown')}\n"
          f"Body:\n{email_data.get('body', 'No content')}\n"
          f"{'-'*60}\n"
        )

    return "\n".join(email_contents) if email_contents else "No emails could be retrieved."

  except HttpError as e:
    tb = traceback.format_exc()
    print("\n🔥 TASK FAILED TRACEBACK 🔥")
    print(tb)
    print("🔥 END TRACEBACK 🔥\n")
    return f"get_emails error: {str(e)}"


@tool
def gmail_send_message(to: str, subject: str, content: str) -> str:
  """
  Send an email using the authenticated user's Gmail account via the Gmail API.

  This tool sends a plain-text email from the currently authenticated Gmail user
  to the specified recipient(s). The 'From' address is automatically set to the
  logged-in user's primary email address.

  Args:
    to (str): The recipient email address(es). Multiple addresses can be
              separated by commas (e.g., "user1@example.com, user2@example.com").
    subject (str): The subject line of the email.
    content (str): The plain-text body content of the email.

  Returns:
    dict: The Gmail API response containing the sent message details
          (including the message ID) if successful, otherwise None.

  Note:
    Requires valid Gmail API credentials with at least the
    'https://www.googleapis.com/auth/gmail.send' scope.
  """
  creds = get_creds(SCOPES)

  try:
    service = build("gmail", "v1", credentials=creds)
    profile = service.users().getProfile(userId="me").execute()

    message = EmailMessage()
    message["To"] = to
    message["From"] = profile["emailAddress"]
    message["Subject"] = subject
    message.set_content(content)

    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    create_message = {"raw": encoded_message}

    send_message = (
      service.users()
      .messages()
      .send(userId="me", body=create_message)
      .execute()
    )
    return f"email sent: {send_message}"
  except HttpError as e:
    tb = traceback.format_exc()
    print("\n🔥 TASK FAILED TRACEBACK 🔥")
    print(tb)
    print("🔥 END TRACEBACK 🔥\n")
    return f"gmail_send_message error: {str(e)}"