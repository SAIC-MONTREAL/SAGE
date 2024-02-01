"""
One Prompt baseline
A single prompt comprised of the user command and the states of all devices is used
The LLM is asked to generate updated states in response. The LLM response should
output the changes that need to be made not the full new state
"""
import os
from dataclasses import dataclass, field
from typing import Type
from copy import deepcopy

import pandas as pd
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

from sage.coordinators.base import CoordinatorConfig, BaseCoordinator
from baselines.templates.oneprompt_templates import (
    oneprompt_prompt_template,
    OnePromptResponse,
)
from sage.testing.fake_requests import db as test_log_db


@dataclass
class OnePromptCoordinatorConfig(CoordinatorConfig):
    """Config to run a demo with OnePrompt baseline."""

    _target: Type = field(default_factory=lambda: OnePromptCoordinator)
    name: str = "OnePrompt"


class OnePromptCoordinator(BaseCoordinator):
    """
    Simple baseline that only uses a single prompt to execute the user command.
    This coordinator only works with testcase evaluations, not real device demos.
    """

    def __init__(self, config: CoordinatorConfig):
        super().__init__(config)

        tv_guide = pd.read_csv(
            os.path.join(os.getenv("SMARTHOME_ROOT"), "sage/testing/tv_guide.csv")
        )
        self.fake_tv_guide = (
            tv_guide[["channel_number", "channel_name", "program_name"]]
            .iloc[1:11]
            .to_dict()
        )

    def execute(self, command: str) -> str:
        """
        Runs the llm  with the provided command
        """

        user_name, command = command.split(":")
        device_state = deepcopy(
            test_log_db.get_device_state(self.config.global_config.test_id)
        )

        parser = PydanticOutputParser(pydantic_object=OnePromptResponse)
        llm_chain = LLMChain(llm=self.llm, prompt=oneprompt_prompt_template)

        planning_answer = llm_chain.run(
            command=command,
            device_state=str(device_state),
            tv_guide=str(self.fake_tv_guide),
        )
        planning_answer = parser.parse(planning_answer)

        if bool(planning_answer.diff):
            if "devices" in planning_answer.diff:
                device_state.update(planning_answer.diff["devices"])
                test_log_db.set_device_state(
                    self.config.global_config.test_id, device_state
                )

        return dict(planning_answer)
