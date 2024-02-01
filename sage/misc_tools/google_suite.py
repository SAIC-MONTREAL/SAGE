"""
GoogleTool: an agent for interacting with Google Calendar and Gmail.
GoogleTool is used as a tool for higher-level agents. It has many subtools to
directly interact with Google services.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type
from datetime import datetime, timezone

from googleapiclient.discovery import Resource
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent_toolkits import GmailToolkit
from langchain.llms.base import BaseLLM
from langchain.tools.gmail.base import GmailBaseTool

from sage.misc_tools.gcloud_auth import gcloud_authenticate
from sage.base import SAGEBaseTool, BaseToolConfig
from sage.utils.llm_utils import LLMConfig


os.environ["CURL_CA_BUNDLE"] = ""


@dataclass
class GmailGetContactsToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: GmailGetContactsTool)
    name: str = "get_contact_list"
    description: str = """Use this tool to return all the user's email contacts.
This is can be used to get the email address of a recipient when composing an email.
Contacts are in the format 'contact name: contact email'."""
    llm_config: LLMConfig = LLMConfig()
    contacts: Dict[str, str] = field(
        default_factory=lambda: {
            "Mom": "mom@example.com",
            "Dad": "dad@example.com",
            "Dmitriy": "d.rivkin@samsung.com",
            "Adam": "sigalsamsung@gmail.com",
            "Amal": "amal.feriani@samsung.com",
        }
    )


class GmailGetContactsTool(SAGEBaseTool):
    llm: BaseLLM = None
    contacts: Dict[str, str] = None

    def setup(self, config: GmailGetContactsToolConfig) -> None:
        self.contacts = config.contacts

    def _run(
        self, **kwargs
    ) -> str:  # kwargs are only here to avoid a crash if the LLM invents inputs
        return str(self.contacts)


class GoogleCalendarCreateEventTool(GmailBaseTool):
    """Tool that creates a Google Calendar event."""

    name: str = "google_calendar_create_event"

    timenow = datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()
    description: str = f"""Use this tool to create an event in Google Calendar.
Inputs are DateTimes `start` and `end`, `summary` (str) which is the title of the event,
and `description` (str) which describes the event. The default duration of events is 1 hour.
DateTimes are given as strings with format of RFC3339 timestamp with time zone offset; current DateTime is
{timenow}"""

    def _run(
        self, start: str, end: str, summary: str, description: str
    ) -> Dict[str, Any]:

        event_result = (
            self.api_resource.events()
            .insert(
                calendarId="primary",
                body={
                    "summary": summary,
                    "description": description,
                    "start": {"dateTime": start, "timeZone": "America/New_York"},
                    "end": {"dateTime": end, "timeZone": "America/New_York"},
                },
            )
            .execute()
        )

        return {
            "id": event_result.get("id", ""),
            "summary": event_result.get("summary", ""),
            "start": event_result["start"]["dateTime"],
            "end": event_result["end"]["dateTime"],
            "description": event_result.get("description", ""),
        }


class GoogleCalendarListEventsTool(GmailBaseTool):
    """Tool that lists Google Calendar events."""

    name: str = "google_calendar_list_events"

    timenow = datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()
    description: str = f"""Use this tool to find events in Google Calendar. The
output is a JSON list of the requested resource. There are 3 optional inputs:
- `max_results` (int), which is the maximum number of results to return (default=10)
- `timeMax` (str), DateTime Upper bound (exclusive) for an event's start time to filter by
- `timeMin` (str), DateTime Lower bound (inclusive) for an event's end time to filter by
DateTimes are given in RFC3339 timestamp with time zone offset;
The current DateTime is {timenow}"""

    def _run(
        self,
        # query: str,
        max_results: int = 10,
        timeMin: str = None,
        timeMax: str = None,
        **kwargs,  # kwargs are only here to avoid a crash if the LLM invents inputs
    ) -> List[Dict[str, Any]]:

        time_range_args = self._get_time_range_args(timeMin, timeMax)

        events = (
            self.api_resource.events()
            .list(
                calendarId="primary",
                # q=query,
                maxResults=max_results,
                singleEvents="True",
                orderBy="startTime",
                **time_range_args,
            )
            .execute()
        )
        ev_list = events.get("items", [])

        trimmed_events = []

        for event in ev_list:
            trimmed_events.append(
                {
                    "id": event.get("id", ""),
                    "summary": event.get("summary", ""),
                    "start": event["start"]["dateTime"],
                    "end": event["end"]["dateTime"],
                    "location": event.get("location", ""),
                    "description": event.get("description", ""),
                }
            )

        return trimmed_events

    def _get_time_range_args(self, timeMin: str, timeMax: str) -> Dict[str, str]:
        """
        Return dict with provided date range args, if supplied.
        We do this so that we only supply date range to api call if we have it,
        otherwise we should avoid specifying a date range altogether.
        """
        time_range_args = {}

        if timeMin is not None:
            time_range_args["timeMin"] = timeMin

        if timeMax is not None:
            time_range_args["timeMax"] = timeMax

        return time_range_args


@dataclass
class GoogleToolConfig(BaseToolConfig):

    _target: Type = field(default_factory=lambda: GoogleTool)
    name: str = "google_tool"
    description: str = """Use this tool to perform actions with user's Gmail and
Google calendar account, and get contacts' names and email addresses.
This tool accepts natural language inputs. Do not specify the user's name in the
query, always assume the name is known. The current user's email address is sagelargelm@gmail.com."""

    llm_config: LLMConfig = None


class GoogleTool(SAGEBaseTool):
    """
    Defines a tool to perform actions with user's Google Calendar and Gmail.
    """

    llm: BaseLLM = None
    gmail_api_resource: Resource = None
    gcal_api_resource: Resource = None

    def setup(self, config: GoogleToolConfig) -> None:
        """Set up the gmail tool"""
        self.gmail_api_resource = gcloud_authenticate(app="gmail")
        self.gcal_api_resource = gcloud_authenticate(app="calendar")
        self.llm = config.llm_config.instantiate()

    def _run(self, text: str) -> str:
        toolkit = GmailToolkit(api_resource=self.gmail_api_resource)
        tools = toolkit.get_tools()
        tools.append(GmailGetContactsToolConfig().instantiate())

        tools.append(GoogleCalendarListEventsTool(api_resource=self.gcal_api_resource))
        tools.append(GoogleCalendarCreateEventTool(api_resource=self.gcal_api_resource))

        agent = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
        )

        try:
            res = agent.run(text)
        except Exception as e:
            res = str(e)

        return res
