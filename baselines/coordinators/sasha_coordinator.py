"""
Sasha coordinator
"""

from dataclasses import dataclass, field
from typing import Type, Any
from copy import deepcopy
from inspect import getsource
import requests

from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

from sage.coordinators.base import CoordinatorConfig, BaseCoordinator
from sage.utils.llm_utils import TGIConfig
from sage.testing.fake_requests import db as test_log_db
from baselines.templates.multistage_sasha_templates import (
    ClarificationResponse,
    FilteringResponse,
    PlanningResponse,
    PrePlanningResponse,
    ReadingResponse,
    PersistentResponse,
)
from baselines.templates.multistage_sasha_templates import (
    clarification_prompt_template,
    filtering_prompt_template,
    planning_prompt_template,
    pre_planning_prompt_template,
    reading_prompt_template,
    persistent_prompt_template,
)


def condition_check(
    trigger_keys: list[str], trigger_value: Any, devices: dict[str, Any]
) -> bool:
    """Checks if a device state satisfies a given condition."""

    if trigger_value == get_key_value(devices, trigger_keys):
        return True

    return False


def get_key_value(dictionary: dict[str, Any], keys: list[str]) -> Any:
    """
    Get the value of a dictionary from the nested list of keys.
    """
    value = dictionary

    for key in keys:
        assert (
            key in value.keys()
        ), f"Invalid keys. Key {key} not found in dictionary keys {value.keys()}"
        value = value[key]

    return value


@dataclass
class SashaCoordinatorConfig(CoordinatorConfig):
    """Config for the Sasha baseline."""

    _target: Type = field(default_factory=lambda: SashaCoordinator)
    name: str = "Sasha"


class SashaCoordinator(BaseCoordinator):
    """
    A baseline implementation of the Sasha paper [1].
    This coordinator only works with testcase evaluations, not real device demos.

    [1] Sasha: creative goal-oriented reasoning in smart homes with large language models
    https://arxiv.org/abs/2305.09802
    """

    def get_chain(self) -> dict[str, Any]:
        """Initialize SASHA chain of LLMs"""
        components = [
            "clarification",
            "filtering",
            "pre_planning",
            "planning",
            "reading",
            "persistent",
        ]
        chain_dict = {key: {} for key in components}
        prompts = [
            clarification_prompt_template,
            filtering_prompt_template,
            pre_planning_prompt_template,
            planning_prompt_template,
            reading_prompt_template,
            persistent_prompt_template,
        ]

        # Add stop tokens for Lemur only
        prompt_list = []

        if isinstance(self.config.llm_config, TGIConfig):
            for p, prompt in enumerate(prompts):
                new_prompt = deepcopy(prompt)
                new_prompt.template = (
                    f"{new_prompt.template}\n Write <FINISHED> after the JSON output."
                )
                prompt_list.append(new_prompt)
        else:
            prompt_list = prompts

        response_list = [
            ClarificationResponse,
            FilteringResponse,
            PrePlanningResponse,
            PlanningResponse,
            ReadingResponse,
            PersistentResponse,
        ]

        for component, prompt, response in zip(components, prompt_list, response_list):
            chain_dict[component]["chain"] = LLMChain(
                llm=self.llm, prompt=prompt, verbose=True
            )
            chain_dict[component]["parser"] = PydanticOutputParser(
                pydantic_object=response
            )

        return chain_dict

    def get_state_descriptions(self, device_state: dict) -> dict:
        """Reduced dimensionality of device state to only include the name of the capabilities (not their values)."""
        new_device_state = {}

        for key in device_state:
            new_device_state[key] = {}

            for key2 in device_state[key]:
                new_device_state[key][key2] = device_state[key][key2].keys()

        return new_device_state

    def execute(self, command: str) -> str:
        """
        Runs the baseline  with the provided command
        """
        _, command = command.split(":")
        chain_dict = self.get_chain()

        device_state = deepcopy(
            test_log_db.get_device_state(self.config.global_config.test_id)
        )

        new_device_state = self.get_state_descriptions(device_state)
        # clarification step
        clarification_answer = chain_dict["clarification"]["chain"].run(
            devices=new_device_state, user_command=command
        )
        clarification_answer = chain_dict["clarification"]["parser"].parse(
            clarification_answer
        )

        if clarification_answer.response == "YES":
            # filtering step
            filtering_answer = chain_dict["filtering"]["chain"].run(
                devices=new_device_state, user_command=command
            )
            filtering_answer = chain_dict["filtering"]["parser"].parse(filtering_answer)
            # pre_planning
            pre_planning_answer = chain_dict["pre_planning"]["chain"].run(
                devices=new_device_state, user_command=command
            )
            pre_planning_answer = chain_dict["pre_planning"]["parser"].parse(
                pre_planning_answer
            )
            device_state_filtered = dict(
                (k, device_state[k])
                for k in filtering_answer.devices
                if k in device_state
            )
            assert pre_planning_answer.response in [
                "control",
                "sensor",
                "persistent",
            ], "Pre-Planning response failed and didn't return Control or Sensor"

            if pre_planning_answer.response == "control":
                # planning step
                planning_answer = chain_dict["planning"]["chain"].run(
                    devices=device_state_filtered, user_command=command
                )
                planning_answer = dict(
                    chain_dict["planning"]["parser"].parse(planning_answer)
                )
                # update state in database

                device_state.update(planning_answer["devices"])
                test_log_db.set_device_state(
                    self.config.global_config.test_id, device_state
                )

                return dict(planning_answer)
            elif pre_planning_answer.response == "sensor":
                # sensor reading prompt
                planning_answer = chain_dict["reading"]["chain"].run(
                    devices=device_state_filtered, user_command=command
                )
                planning_answer = chain_dict["reading"]["parser"].parse(planning_answer)

                return dict(planning_answer)
            elif pre_planning_answer.response == "persistent":
                planning_answer = chain_dict["persistent"]["chain"].run(
                    devices=device_state_filtered, user_command=command
                )
                planning_answer = chain_dict["persistent"]["parser"].parse(
                    planning_answer
                )

                return self.add_condition(planning_answer, device_state_filtered)
        else:
            return {"output": clarification_answer.explanation}

    def add_condition(self, command: str, devices: dict[str, Any]) -> str:
        """Defines the condition checking code for persistent commands and adds it to the trigger server."""
        code_define = f"""
from testing.fake_requests import db
from typing import Dict, List
devices = db.get_device_state('{self.config.global_config.test_id}')
test_id = '{self.config.global_config.test_id}'
trigger_keys = {command.trigger}
trigger_value='{command.trigger_value}'
        """

        code_define += "\n"
        code_define += getsource(get_key_value)
        code_define += "\n"
        code_define += getsource(condition_check)

        fn_name = "condition_check"
        code_registry = {}
        last_result = condition_check(
            trigger_keys=command.trigger,
            trigger_value=command.trigger_value,
            devices=devices,
        )
        code_registry[fn_name] = {
            "code_define": code_define,
            "code_run": "condition_check(trigger_keys, trigger_value, devices)",
            "last_result": last_result,
        }

        info = {}
        info["user_name"] = "None"
        info["function_name"] = fn_name
        info["action_description"] = command.output
        info["notify_when"] = True
        requests.post(
            self.config.global_config.condition_server_url + "/add_condition",
            json={"code": {fn_name: code_registry[fn_name]}, "condition": info},
        )
        return command.output
