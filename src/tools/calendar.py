import traceback
from datetime import datetime, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain.tools import tool
from .gmail import get_creds

from datetime import timedelta

def _hkt_to_utc(dt_str: str) -> str:
    """
    Convert a naive ISO 8601 datetime string interpreted as HKT
    into UTC RFC3339 (Z).

    IMPORTANT:
    - Input MUST be naive (no timezone, no 'Z')
    - Input is assumed to be Hong Kong Time (UTC+8)
    """
    dt = datetime.fromisoformat(dt_str.replace("Z", ""))

    # Assume HKT if naive
    if dt.tzinfo is None:
        dt = dt - timedelta(hours=8)

    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

@tool
def search_calendar_events(
  time_min: str, 
  time_max: str,
  query: str | None = None,
  calendar_id: str = "primary",
  max_results: int = 20,
) -> str:
  """
  Search events containing a keyword or phrase (similar to Gmail search).

  Args:
    query (str, optional): 
      Text to search across event title, description, location, and attendees.
      Examples: "meeting", "birthday", "team lunch", "project deadline"
      Leave empty or None to return ALL events in the time range.

    calendar_id (str): 
      "primary" for your main calendar, or a specific calendar ID.

    time_min (str): 
      (ISO 8601 datetime string in HKT (naive, no Z))
      Lower bound for event start time.
      Use this to filter events starting AFTER this time.

    time_max (str): 
      (ISO 8601 datetime string in HKT (naive, no Z))
      Upper bound for event end time.
      Use this to filter events ending BEFORE this time.

    max_results (int): Maximum number of events to return (default 20).

  Returns:
    Formatted list of matching events.
  """
  creds = get_creds()
  try:
    service = build("calendar", "v3", credentials=creds)

    params = {
      "calendarId": calendar_id,
      "q": query,
      "maxResults": max_results,
      "singleEvents": True,
      "orderBy": "startTime",
    }
    if time_min:
      params["timeMin"] = _hkt_to_utc(time_min)
    if time_max:
      params["timeMax"] = _hkt_to_utc(time_max)

    events_result = service.events().list(**params).execute()
    events = events_result.get("items", [])

    if not events:
      return f"No events found matching query: '{query}'"

    result = []
    for event in events:
      start = event["start"].get("dateTime", event["start"].get("date"))
      end = event["end"].get("dateTime", event["end"].get("date"))
      result.append(
        f"ID: {event['id']}\n"
        f"Summary: {event.get('summary', 'No title')}\n"
        f"Start: {start}\n"
        f"End: {end}\n"
        f"Location: {event.get('location', 'None')}\n"
        f"Description: {event.get('description', 'None')[:500]}\n"
        f"Link: {event.get('htmlLink', 'N/A')}\n"
        f"{'-'*60}"
      )
  
    return "\n".join(result)
  except HttpError as e:
    tb = traceback.format_exc()
    print("\n🔥 TASK FAILED TRACEBACK 🔥\n", tb, "\n🔥 END TRACEBACK 🔥\n")
    return f"search_calendar_events error: {str(e)}"
  
@tool
def create_calendar_event(
  summary: str,
  start_time: str,   
  end_time: str,  
  calendar_id: str = "primary",
  description: str = None,
  location: str = None,
  attendees: list[str] = None, 
) -> str:
  """
  Create a new event.

  Args:
    summary: Title of the event
    start_time: ISO 8601 datetime string in Hong Kong Time (HKT),
      WITHOUT timezone or 'Z' (e.g. "2026-01-10T20:00:00"),
      OR date string "YYYY-MM-DD" for all-day events.
    end_time: Same as start_time
    calendar_id: Target calendar
    description: Optional details
    location: Optional
    attendees: Optional list of email addresses

  Returns:
      Success message with link and event ID.
  """
  creds = get_creds()
  try:
    service = build("calendar", "v3", credentials=creds)

    event = {
      "summary": summary,
      "description": description,
      "location": location,
      "start": {},
      "end": {},
    }

    # Detect all-day vs timed
    if "T" in start_time:
      event["start"] = {"dateTime": _hkt_to_utc(start_time), "timeZone": "UTC"}
      event["end"] = {"dateTime": _hkt_to_utc(end_time), "timeZone": "UTC"}
    else:
      event["start"] = {"date": start_time}
      event["end"] = {"date": end_time}

    if attendees:
      event["attendees"] = [{"email": email} for email in attendees]

    created_event = service.events().insert(calendarId=calendar_id, body=event).execute()

    return f"Event created successfully!\nTitle: {summary}\nLink: {created_event.get('htmlLink')}\nID: {created_event['id']}"

  except HttpError as e:
    tb = traceback.format_exc()
    print("\n🔥 TASK FAILED TRACEBACK 🔥\n", tb, "\n🔥 END TRACEBACK 🔥\n")
    return f"create_calendar_event error: {str(e)}"

@tool
def update_calendar_event(
  event_id: str,
  calendar_id: str = "primary",
  summary: str = None,
  start_time: str = None,
  end_time: str = None,
  description: str = None,
  location: str = None,
) -> str:
  """
  Update an existing event. Only provide fields you want to change.

  Args:
    event_id: The event ID (get from search or list tools)
    calendar_id: Calendar containing the event
    summary, start_time, end_time, etc.: New values

  Returns:
      Success message or error.
  """
  creds = get_creds()
  try:
    service = build("calendar", "v3", credentials=creds)
    event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()

    if summary:
      event["summary"] = summary
    if description:
      event["description"] = description
    if location:
      event["location"] = location

    if start_time or end_time:
      if start_time and "T" in start_time:
        event["start"] = {"dateTime": _hkt_to_utc(start_time), "timeZone": "UTC"}
      elif start_time:
        event["start"] = {"date": start_time}

      if end_time and "T" in end_time:
        event["end"] = {"dateTime": _hkt_to_utc(end_time), "timeZone": "UTC"}
      elif end_time:
        event["end"] = {"date": end_time}

    updated_event = service.events().update(calendarId=calendar_id, eventId=event_id, body=event).execute()

    return f"Event updated successfully!\nTitle: {updated_event.get('summary')}\nLink: {updated_event.get('htmlLink')}"

  except HttpError as e:
    tb = traceback.format_exc()
    print("\n🔥 TASK FAILED TRACEBACK 🔥\n", tb, "\n🔥 END TRACEBACK 🔥\n")
    return f"update_calendar_event error: {str(e)}"


@tool
def delete_calendar_event(event_id: str, calendar_id: str = "primary") -> str:
  """
  Delete an event permanently.

  Args:
    event_id: Event ID to delete
    calendar_id: Calendar containing the event

  Returns:
    Confirmation message.
  """
  creds = get_creds()
  try:
    service = build("calendar", "v3", credentials=creds)
    service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
    return f"Event {event_id} deleted successfully."

  except HttpError as e:
    tb = traceback.format_exc()
    print("\n🔥 TASK FAILED TRACEBACK 🔥\n", tb, "\n🔥 END TRACEBACK 🔥\n")
    return f"delete_calendar_event error: {str(e)}"