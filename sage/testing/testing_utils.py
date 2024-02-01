"""Testing helper functions"""
import html
import inspect
import os
import pickle as pkl
import time
import uuid
from functools import lru_cache
from typing import Any

import requests
from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel
from langchain.pydantic_v1 import Field

from sage.base import BaseConfig
from sage.coordinators.base import BaseCoordinator
from sage.coordinators.base import CoordinatorConfig
from sage.testing.fake_requests import db
from sage.testing.fake_requests import set_test_id


current_save_dir = [None]


# helper methods
def setup(
    device_state: dict[str, Any], coord_config: CoordinatorConfig = CoordinatorConfig
) -> tuple[str, BaseCoordinator]:
    """Setup the env to run a specific testcase"""
    # generate test id
    test_id = str(uuid.uuid4())
    # set the global test id in the module and in the global config object
    set_test_id(test_id)

    # make a new coordinator every time so that memory resets
    config = coord_config
    config.global_config.test_id = test_id
    # pick up name of caller
    caller_name = inspect.currentframe().f_back.f_code.co_name
    config.global_config.logpath = current_save_dir[0].joinpath(caller_name)
    coordinator = config.instantiate()

    # write state to db
    db.set_device_state(test_id, device_state)

    # reset condition server
    requests.get(BaseConfig.global_config.condition_server_url + "/reset")
    # return test id

    return test_id, coordinator


def listen(demo_config: BaseConfig, timeout: int = 15) -> None:
    """Listen for responses from the trigger server"""
    # timeout in seconds
    # poll triggers
    start_time = time.time()

    while (start_time + timeout) >= time.time():
        for _, url in demo_config.trigger_servers:
            trigger = requests.get(url + "/check_triggers").json()
            print("got trigger from %s" % url, trigger)

            if trigger:
                return trigger["user"], trigger["command"]
        time.sleep(1)


@lru_cache(maxsize=None)
def get_base_device_state():
    """Loads a device state from a file"""
    # load it from a pickle
    with open(
        os.getenv("SMARTHOME_ROOT") + "/sage/testing/device_state0.pkl", "rb"
    ) as f:
        state = pkl.load(f)

    return {s[0]["device_id"]: s[0]["components"] for s in state}


@lru_cache(maxsize=None)
def get_min_device_state():
    """
    Loads a reduced version of the device state from a file
    Used to avoid exceeding the maximum number of tokens
    """
    # load it from a pickle
    with open(
        os.getenv("SMARTHOME_ROOT") + "/sage/testing/device_state_4383.pkl", "rb"
    ) as f:
        state = pkl.load(f)

    return {s[0]["device_id"]: s[0]["components"] for s in state}


class EvaluationResponse(BaseModel):
    output: str = Field(
        description="Contains a single word, either YES or NO depending on if answers match or not."
    )


evaluation_template = """
You are an an expert that evaluates if two answers match (i.e. are similar / equivalent) or do not match (are not similar / not equivalent).

Answer 1: {answer_1}
Answer 2: {answer_2}

{format_instructions}
"""
# Assign appropriate settings to the relevant devices.
# Your response should be a JSON of all of the changed device states.

# input template to llm
evaluation_prompt_template = PromptTemplate(
    template=evaluation_template,
    input_variables=["answer_1", "answer_2"],
    partial_variables={
        "format_instructions": PydanticOutputParser(
            pydantic_object=EvaluationResponse
        ).get_format_instructions()
    },
)


def manual_gmail_search(api_resource, query, maxResults=10):
    """
    Search gmail directly using python. Used to check test execution. Add any dates
    to be searched to the query, if wanted.
    Returns a list of dicts each representing an email.
    """
    result = (
        api_resource.users()
        .messages()
        .list(userId="me", q=query, maxResults=maxResults)
        .execute()
    )
    assert (
        result.get("messages") is not None
    ), f"No emails were found with the manual verification query '{query}', the LLM system likely failed to perform the desired task."

    message_ids = [item["id"] for item in result["messages"]]

    messages = []
    for message_id in message_ids:
        curr = {"id": message_id}
        message = (
            api_resource.users().messages().get(userId="me", id=message_id).execute()
        )
        curr["snippet"] = html.unescape(message["snippet"])

        for d in message["payload"]["headers"]:
            if d["name"] == "Subject":
                curr["subject"] = d["value"]
            if d["name"] == "To":
                curr["recipient"] = d["value"]
            if d["name"] == "From":
                curr["sender"] = d["value"]
            if d["name"] == "Date":
                curr["date"] = d["value"]
        messages.append(curr)

    return messages


def pretty_print_email(messages_info: list[dict[str, str]]) -> str:
    """
    Return strings of email info retrieved in manual_gmail_search. The input
    to this function should be the output of manual_gmail_search.
    """
    text = ""
    for i, message in enumerate(messages_info):
        text += f"""
Email {i+1}:
From: {message['sender']}
Recipient: {message['recipient']}
Subject: {message['subject']}
Snippet of email body:
{message['snippet']}
"""
    return text
