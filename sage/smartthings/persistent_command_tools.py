"""
Tools for dealing with persistent commands.
"""
import traceback
from dataclasses import dataclass
from dataclasses import field
from typing import Type

import requests
from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.tools import BaseTool
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage

from sage.base import BaseConfig
from sage.base import BaseToolConfig
from sage.base import SAGEBaseTool
from sage.smartthings.device_disambiguation import DeviceDisambiguationToolConfig
from sage.smartthings.smartthings_tool import ApiDocRetrievalToolConfig
from sage.smartthings.smartthings_tool import SmartThingsPlannerToolConfig
from sage.testing.fake_requests import replace_requests_with_fake_requests
from sage.utils.common import parse_json
from sage.utils.llm_utils import LLMConfig
from sage.utils.logging_utils import get_callback_handlers
from sage.utils.trigger_server import run_code


@dataclass
class PythonInterpreterToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: PythonInterpreterTool)
    name: str = "python_interpreter"
    description: str = """Use this tool to run python code to automate tasks when necessary.
Use like python_interpreter(<python code string here>). Make sure the last line is the entrypoint to your code."""


code_registry = {}


class PythonInterpreterTool(SAGEBaseTool):
    test_id: str = None

    def setup(self, config: PythonInterpreterToolConfig) -> None:
        """Setup the tool"""
        self.test_id = config.global_config.test_id

    def _run(self, command) -> str:
        # I don't know if GPT always writes code where the last line is the actual
        # call to the code, but let's assume for now it is, if not, can always
        # do some prompt engineering to make it so.
        # some thing it likes to put in that break stuff
        command = command.replace("```", "").replace("python", "")

        if self.test_id is not None:
            command = replace_requests_with_fake_requests(command, self.test_id)
        try:
            code_define, code_run = command.strip("\n").rsplit("\n", 1)

            if code_run[:5] == "print":
                code_run = code_run[6:-1]
            # add this stuff to the registry so we can refer to it later
            fn_name = (
                code_run.split("(")[0]
                .replace("\n", "")
                .replace(" ", "")
                .replace("\t", "")
            )
            result = run_code(code_define, code_run)
            code_registry[fn_name] = {
                "code_define": code_define,
                "code_run": code_run,
                "last_result": result,
            }

            return result
        except Exception:
            return traceback.format_exc()


@dataclass
class ConditionCheckerToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: ConditionCheckerTool)
    name: str = "condition_checker_tool"
    description: str = """
Use this tool to check whether a certain condition is satisfied. Accepts natural language commands. Returns the name of a function which checks the condition. Inputs should be phrased as questions. In addition to the question that needs to be checked, you should also provide any extra information that might contextualize the question.
"""
    llm_config: LLMConfig = None
    tool_configs: tuple[BaseConfig, ...] = (
        SmartThingsPlannerToolConfig(),
        ApiDocRetrievalToolConfig(),
        DeviceDisambiguationToolConfig(),
        PythonInterpreterToolConfig(),
    )


def create_and_run_condition_codewriter(
    command: str, llm: BaseChatModel, logpath: str, tools: list[BaseTool]
) -> str:
    tool_names = ", ".join([tool.name for tool in tools])
    tool_descriptions = "\n".join(
        [f"{tool.name}: {tool.description}" for tool in tools]
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""
You are an agent that writes code that checks whether a condition is satisfied by querying some API.

You should first plan the sequence of API calls you need to make. Then you should get detailed
documentation for each of the API calls. Finally, you should write python code to check the condition. Remember, the code:

- should be tested using a python interpreter
- should involve the checking of a single condition that involves checking the state of a single device that best fulfills the user request
- should follow the correct dictionary key structure (e.g. data['components']['main']['capability_name']['command_name']['value'])
- the function must return True if the device state condition is satisfied and False otherwise
- the code should only include a single function definition that does not include any arguments.
- the last line of the programe should be an example of the function call

If the code doesn't work, you should try to figure out why and fix it. Your final answer should include the current status of the condition and the name of the function you wrote to check it. Remember, the function should return True or False.

The API you are primarily going to be using is the smartthings API. The token is: "bb06de1d-e45a-4449-be5d-b8bb588222dd"
You may also use free APIs if know of any, but you will not be able to search for their documentation.
The only documentation you can search for is related to the smartthings API.

Here are the tools to plan requests:
{tool_descriptions}

Starting below, you should follow this format:

User query: the query a User wants help with related to the API
Thought: you should always think about what to do
Action: the action to take, should be one of the tools [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I am finished executing a plan and have the information the user asked for or the data the user asked to create
Final Answer: the final output from executing the plan. This must always include the name of the function
you wrote to check the condition.

Your must always output a thought, action, and action input.

Begin!
"""
            ),
            HumanMessagePromptTemplate.from_template(
                """
Condition to check: {input}
Thought: I should generate a plan to help with checking this condition.
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
        handle_parsing_errors=True,
        callbacks=callbacks,
    )
    return agent_executor.run(command)


class ConditionCheckerTool(SAGEBaseTool):
    llm: BaseChatModel = None
    logpath: str = None

    def setup(self, config: ConditionCheckerToolConfig):
        self.llm = config.llm_config.instantiate()
        self.logpath = config.global_config.logpath

    def _run(self, command: str) -> str:
        return create_and_run_condition_codewriter(
            command, self.llm, self.logpath, self.tools
        )


condition_registry = []


@dataclass
class NotifyOnConditionToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: NotifyOnConditionTool)
    name: str = "notify_on_condition_tool"
    description: str = """
Use this tool to get notified when a condition occurs. It should be called with a json string with keys:
function_name (str): name of the function that checks the condition
notify_when (bool): [true, false]
condition_description (str)
action_description (str): what to do when the condition is met
user_name (str)
"""


class NotifyOnConditionTool(SAGEBaseTool):
    server_url: str = None

    def setup(self, config: NotifyOnConditionToolConfig):
        self.server_url = config.global_config.condition_server_url

    def _run(self, command: str) -> str:

        info = parse_json(command)
        if info is None:
            return "Invalid input format. Input to the notify_on_condition_tool should be a json string with 5 keys: function_name (str), notify_when (bool), condition_description (str), action_description (str) and user_name (str)."

        fn_name = info["function_name"]
        if fn_name not in code_registry:
            return "Unknown function: " + info["function_name"]

        requests.post(
            self.server_url + "/add_condition",
            json={"code": {fn_name: code_registry[fn_name]}, "condition": info},
        )
        # condition_registry.append(info)
        return "You will be notified when the condition occurs."
