"""
An agent for interaction with SmartThings devices.

Used as a tool SAGE.
"""
import json
from dataclasses import dataclass
from dataclasses import field
from difflib import SequenceMatcher
from typing import Any, Dict
from typing import List
from typing import Type

import numpy as np
import requests
from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.tools import BaseTool
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage

import sage.testing.fake_requests
from sage.base import BaseConfig
from sage.base import BaseToolConfig
from sage.base import SAGEBaseTool
from sage.smartthings.device_disambiguation import DeviceDisambiguationToolConfig
from sage.smartthings.docmanager import DocManager
from sage.utils.common import parse_json
from sage.utils.llm_utils import LLMConfig
from sage.utils.llm_utils import TGIConfig
from sage.utils.logging_utils import get_callback_handlers


def most_similar_id(device, all_devices):
    sims = [SequenceMatcher(None, device, d).ratio() for d in all_devices]
    idx = np.argmax(sims)
    closest_sim = sims[idx]

    if closest_sim > 0.5:
        return all_devices[idx]

    return "to use the planner to figure out the right ID"


@dataclass
class GetAttributeToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: GetAttributeTool)
    name: str = "get_attribute"
    description: str = """
Use this to get an attribute from a device capability.
Input to the tool should be a json string with 4 keys:
device_id (str)
component (str)
capability (str)
attribute (str)
"""


class GetAttributeTool(SAGEBaseTool):
    """
    Tool for getting a smartthings attribute.
    """

    dm: DocManager = None
    requests_module: Any = requests
    smartthings_token: str = None

    def setup(self, config: GetAttributeToolConfig):
        if config.global_config.test_id is not None:
            self.requests_module = sage.testing.fake_requests
        self.smartthings_token = config.global_config.smartthings_token
        self.dm = DocManager.from_json(config.global_config.docmanager_cache_path)

    def _run(self, text: str):

        attr_spec = parse_json(text)
        if attr_spec is None:
            return "Invalid input format. Input to the get_attribute tool should be a json string with 4 keys: device_id (str), component (str), capability (str), attribute (str)"

        if isinstance(attr_spec, list):
            if len(attr_spec) == 1:
                attr_spec = attr_spec[0]
            else:
                return "Invalid usage: input to this tool should be a dict, not a list"

        device_id = attr_spec["device_id"]
        component = attr_spec["component"]
        capability = attr_spec["capability"]

        if device_id not in self.dm.default_devices:
            return (
                "The device ID you specified does not exist. Did you mean %s?"
                % most_similar_id(device_id, self.dm.default_devices)
            )

        headers = {"Authorization": "Bearer %s" % self.smartthings_token}
        if self.dm.has_refresh_capability(device_id):
            post_url = f"https://api.smartthings.com/v1/devices/{device_id}/commands"
            body = {
                "commands": [
                    {
                        "component": "main",
                        "capability": "refresh",
                        "command": "refresh",
                        "arguments": [],
                    }
                ]
            }
            self.requests_module.post(url=post_url, json=body, headers=headers)

        get_url = f"https://api.smartthings.com/v1/devices/{device_id}/components/{component}/capabilities/{capability}/status"
        response = self.requests_module.get(get_url, headers=headers)
        if response.status_code != 200:
            return (
                json.dumps(response.json())
                + ". Check the API documentation using the ApiDocRetrievalTool tool."
            )

        try:
            response = self.requests_module.get(get_url, headers=headers)
            resp = response.json()
            if isinstance(resp, dict) and attr_spec["attribute"] in resp:
                return resp[attr_spec["attribute"]]
            return resp
        except Exception:
            return "Attribute not available. Check the API documentation using the ApiDocRetrievalTool tool."

    def _arun(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class ExecuteCommandToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: ExecuteCommandTool)
    name: str = "execute_command"
    description: str = """
Use this to execute a command on a device. When executing a command, make sure you have the right arguments.
Input to the tool should be a json string with 5 keys:
device_id (str)
component (str)
capability (str)
command (str)
args (list)
"""


class ExecuteCommandTool(SAGEBaseTool):
    """
    Tool for executing a smartthings command.
    """

    requests_module: Any = requests
    dm: DocManager = None
    smartthings_token: str = None

    def setup(self, config: ExecuteCommandToolConfig):
        if config.global_config.test_id is not None:
            self.requests_module = sage.testing.fake_requests
        self.smartthings_token = config.global_config.smartthings_token
        self.dm = DocManager.from_json(config.global_config.docmanager_cache_path)

    def _run(self, text: str):
        exec_spec = parse_json(text)

        if exec_spec is None:
            return "Invalid input format. Input to the execute_command tool should be a json string with 5 keys: device_id (str), component (str), capability (str), command (str) and args (list)."

        if isinstance(exec_spec, list):
            if len(exec_spec) == 1:
                exec_spec = exec_spec[0]
            else:
                return "Invalid usage: input to this tool should be a dict, not a list"

        device_id = exec_spec["device_id"]
        if device_id not in self.dm.default_devices:
            return (
                "The device ID you specified does not exist. Did you mean %s?"
                % most_similar_id(device_id, self.dm.default_devices)
            )

        post_url = f"https://api.smartthings.com/v1/devices/{device_id}/commands"
        headers = {"Authorization": "Bearer %s" % self.smartthings_token}

        body = {
            "commands": [
                {
                    "component": exec_spec["component"],
                    "capability": exec_spec["capability"],
                    "command": exec_spec["command"],
                    "arguments": exec_spec.get("args", []),
                }
            ]
        }
        response = self.requests_module.post(url=post_url, json=body, headers=headers)
        if response.status_code != 200:
            return (
                json.dumps(response.json())
                + ". Check the API documentation for more information using the ApiDocRetrievalTool tool."
            )

        return json.dumps(
            self.requests_module.post(url=post_url, json=body, headers=headers).json()
        )

    def _arun(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class ApiDocRetrievalToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: ApiDocRetrievalTool)
    name: str = "api_doc_retrieval"
    description: str = """
Use this to check the API documentation for specific components and capabilities of different devices.
Using this tool before interacting with the API will increase your chance of success.
Input to the tool should be a json string comprising a list of dictionaries.
Each dictionary in the list should contain two keys:
device_id (guid string)
capability_id (str)
"""


class ApiDocRetrievalTool(SAGEBaseTool):
    dm: DocManager = None

    def setup(self, config: ApiDocRetrievalToolConfig) -> None:
        self.dm = DocManager.from_json(config.global_config.docmanager_cache_path)

    def _run(self, text):

        spec = parse_json(text)
        if spec is None:
            return "Invalid input format. Input to the api_doc_retrieval tool should be a json string comprising a list of dictionaries. Each dictionary in the list should contain two keys: device_id (guid string) and capability_id (str)."

        device_cap_strings = []
        if not isinstance(spec, list):
            return "Invalid input format. Make sure that the input is a json string comprising a list of dictionaries. Each dictionary in the list should contain two keys: device_id (guid string) and capability_id (str)."

        for obj in spec:
            if "device_id" not in obj.keys():
                return "Invalid input format. Make sure that the input is a json string comprising a list of dictionaries. Each dictionary in the list should contain two keys: device_id (guid string) and capability_id (str)."

            if obj["device_id"] not in self.dm.default_devices:
                device_cap_strings.append(
                    "The device ID you specified does not exist. Did you mean %s?"
                    % most_similar_id(obj["device_id"], self.dm.default_devices)
                )
                continue

            found_cap = False
            for device_cap in self.dm.device_capabilities[obj["device_id"]]:
                if device_cap["capability_id"] == obj["capability_id"]:
                    found_cap = True
                    break

            if not found_cap:
                device_cap_strings.append(
                    "The device %s does not have capability %s."
                    % (obj["device_id"], obj["capability_id"])
                )
                continue

            device_cap_strings.append(
                self.dm.device_capability_details(
                    obj["device_id"], obj["capability_id"]
                )
            )
        device_cap_string = "\n".join(device_cap_strings)
        return device_cap_string

    def _arun(self, *args, **kwargs):
        raise NotImplementedError


def create_and_run_smartthings_agent_v2(
    command: str, llm: BaseChatModel, logpath: str, tools: List[BaseTool]
) -> str:
    tool_names = ", ".join([tool.name for tool in tools])
    tool_descriptions = "\n".join(
        [f"{tool.name}: {tool.description}" for tool in tools]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""
You are an agent that assists with queries against some API.

Instructions:
- Include a description of what you've done in the final answer, include device IDs
- Use channel numbers ONLY when reading or manipulating TV content.
- If you encounter an error, try to think what is the best action to solve the error instead of trial and error.

Here are the tools to plan and execute API requests:
{tool_descriptions}

Starting below, you should follow this format:

User query: the query a User wants help with related to the API
Thought: you should always think about what to do
Action: the action to take, should be one of the tools [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I am finished executing a plan and have the information the user asked for or the data the user asked to create
Final Answer: the final output from executing the plan. Add <FINISHED> after your final answer.

Your must always output a thought, action, and action input.
Do not forget to say I'm finished when the user's command is executed.
Begin!
"""
            ),
            HumanMessagePromptTemplate.from_template(
                """
User query: {input}
Thought: I should generate a plan to help with this query.
{agent_scratchpad}
""",
                input_variables=["input", "agent_scratchpad"],
            ),
        ],
    )

    callbacks = get_callback_handlers(logpath)
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt, callbacks=callbacks, verbose=True),
        allowed_tools=[tool.name for tool in tools],
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        callbacks=callbacks,
        handle_parsing_errors=True,
    )
    return agent_executor.run(command)


@dataclass
class SmartThingsPlannerToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: SmartThingsPlannerTool)
    name: str = "api_usage_planner"
    description: str = "Used to generate a plan of api calls to make to execute a command. The input to this tool is the user command in natural language. Always share the original user command to this tool to provide the overall context."
    llm_config: LLMConfig = None


class SmartThingsPlannerTool(SAGEBaseTool):
    chain: LLMChain = None
    logpath: str = None

    def setup(self, config: SmartThingsPlannerToolConfig):
        if isinstance(config.llm_config, TGIConfig):
            config.llm_config = TGIConfig(stop_sequences=["Human", "<FINISHED>"])
        llm = config.llm_config.instantiate()
        self.logpath = config.global_config.logpath
        dm = DocManager.from_json(config.global_config.docmanager_cache_path)
        (
            one_liners_string,
            device_capability_string,
        ) = dm.capability_summary_for_devices()
        prompt = ChatPromptTemplate.from_messages(
            [
                # - Restate the query 3 different ways
                SystemMessage(
                    content=f"""
You are a planner that helps users interact with their smart devices.
You are given a list of high level summaries of device capabilities ("all capabilities:").
You are also given a list of available devices ("devices you can use") which will tell you the name and device ID of the device, as well as listing which capabilities the device has.
Your job is to figure out the sequence of which devices and capabilities to use in order to execute the user's command.

Follow these instructions:
- Include device IDs (guid strings), capability ids, and explanations of what needs to be done in your plan.
- It is often unclear exactly which device / devices the user is referring to. If you don't know exactly which device to use, list all of the device IDs. Try to come-up with a disambiguation strategy, otherwise explain that further disambiguation is required.
- The capability information you receive is not detailed. Often there will be multiple capabilities that sound like they might work. You should list all of the ones that might work to be safe.
- Don't always assume the devices are already on.
- Some devices can have more than one components. If using one results in a failure, try another one.

all capabilities:
{one_liners_string}

devices you can use:
{device_capability_string}

Use the following format:
Device Ids: list of relevant devices IDs and names
Capabilities: list of relevant capabilities
Plan: steps to execute the command
Explanation: Any further explanations and notes
<FINISHED>
"""
                ),
                HumanMessagePromptTemplate.from_template(
                    "{query}.",
                    input_variables=["query"],
                ),
            ],
        )

        callbacks = get_callback_handlers(self.logpath)
        self.chain = LLMChain(llm=llm, prompt=prompt, callbacks=callbacks, verbose=True)

    def _run(self, command) -> str:
        # try:
        #    json.loads(command)
        # The LLM fails to give the command in natural language
        #    return "The command should be in natural language and not a json."
        # except json.decoder.JSONDecodeError:
        return self.chain.run(command)


@dataclass
class SmartThingsToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: SmartThingsTool)
    name: str = "smartthings_tool"
    description: str = """
Use this to interact with smartthings. Accepts natural language commands. Do not omit any details from the original command. Use this tool to determine which device can accomplish the query.
"""
    llm_config: LLMConfig = None
    tool_configs: tuple[BaseConfig, ...] = (
        SmartThingsPlannerToolConfig(),
        ApiDocRetrievalToolConfig(),
        GetAttributeToolConfig(),
        ExecuteCommandToolConfig(),
        DeviceDisambiguationToolConfig(),
    )


class SmartThingsTool(SAGEBaseTool):
    llm: BaseChatModel = None
    logpath: str = None

    def setup(self, config: SmartThingsToolConfig):
        if isinstance(config.llm_config, TGIConfig):
            config.llm_config = TGIConfig(stop_sequences=["Human", "<FINISHED>"])

        self.llm = config.llm_config.instantiate()
        self.logpath = config.global_config.logpath

    def _run(self, command) -> Any:
        return create_and_run_smartthings_agent_v2(
            command, self.llm, self.logpath, self.tools
        )
