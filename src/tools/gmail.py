import os.path
import base64
import re
from email import policy
from email.parser import BytesParser
from typing import Literal
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain.tools import tool
from ..utils.helper import clean_html_content

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def get_creds():
  creds = None
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
        "credentials.json", SCOPES
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
  
  except HttpError as error:
    # TODO(developer) - Handle errors from gmail API.
    print(f"An error occurred: {error}")

@tool
def get_emails(
  labelIds: Literal["INBOX", "UNREAD", "STARRED", "IMPORTANT"] = "INBOX",
  maxResults: int = 5,
  full_messages_str: bool = True
): 
  """
  Fetch emails from Gmail.
  
  Args:
    labelIds: The label to filter emails by. Can be "INBOX", "UNREAD", 
              "STARRED", or "IMPORTANT". Defaults to "INBOX".
    maxResults: Maximum number of emails to retrieve. Defaults to 5.
    full_messages_str: If True, returns full email content as a string. 
                      If False, returns basic message metadata.
  
  Returns:
    If full_messages_str is True: A string containing concatenated email contents.
    If full_messages_str is False: A list of message metadata dictionaries.
      
  Example:
    get_emails(labelIds="INBOX", maxResults=3)  # Gets 3 latest inbox emails
    get_emails(labelIds="UNREAD", full_messages_str=True)  # Gets unread emails with full content
  """
  print(
    f"Running..."
    f"labelIds: {labelIds}"
    f"maxResults: {maxResults}"
    f"full_messages_str: {full_messages_str}"
  )

  creds = get_creds()
  try:
    # Call the Gmail API
    service = build("gmail", "v1", credentials=creds)
    results = (
      service.users().messages().list(
        userId="me", 
        labelIds=[labelIds], 
        maxResults=maxResults,
        includeSpamTrash=False
      ).execute()
    )
    messages = results.get("messages", [])

    if not messages:
      print("No messages found.")
      return

    if full_messages_str:
      email_contents = []
      for message in messages:
        email_data = get_email(service, message["id"])
        if email_data:
          email_contents.append(
            f"Subject: {email_data.get('subject', 'No Subject')}\n"
            f"From: {email_data.get('from', 'Unknown')}\n"
            f"Date: {email_data.get('date', 'Unknown')}\n"
            f"Body: {email_data.get('body', 'No content')}\n"
          )
      return "\n\n".join(email_contents) if email_contents else "No emails could be retrieved."

    else:
      return messages

  except HttpError as error:
    # TODO(developer) - Handle errors from gmail API.
    print(f"An error occurred: {error}")