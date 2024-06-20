"""
Run through test cases.

These tests are set up not to interact with the real system, which makes them more
reliable and convenient. This does mean that some amount of hackery was necessary to
make the agent think it's talking to devices when it's really just interacting with
an object in memory. This in-memory object is generated simply by dumping the state
of the devices at some point in time (done through smartthings/db.py __main__ block).
Most of this hackery happens in testing/fake_requests.py. This module is intended as
a drop-in replacement for python's requests.py. Traffic to the smartthings API is
intercepted and used to effect changes to devices state. The device state is tracked
for each test separately in the Mongo DB (testing.fake_requests.db), and state updates
are done by writing the state to this DB. A DB is used to support simultaneous access
from multiple processes, which is necessary because the state is read not only by the
agent but also by the code written by the agent, which is executed by the polling server.
By the way, the need to intercept requests made by the python code written by the LLM
is what motivated me to create the fake requests module, as opposed to intercepting
the traffic at a higher level (e.g. in the GetAttribute and ExecuteCommand tools).
When in testing mode, the code written by the LLM which does "import requests" is
modified to "import testing.fake_requests as requests; requests.set_test_id(<test_id>)".
This injection allows the requests in the LLM-written code access to be routed to the
fake object.

A few gotchas:
- We have to write custom code to handle each different command (in the function
testing.fake_requests.request)
- The current test ID is set globally in the fake_requests module. This means that you should NOT
run tests using thread concurrency (but process concurrency should be OK).
"""
import json
import os
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import tyro
import yaml

from sage.base import BaseConfig
from sage.base import GlobalConfig
from baselines.coordinators.oneprompt_coordinator import OnePromptCoordinatorConfig
from sage.coordinators.sage_coordinator import SAGECoordinatorConfig
from baselines.coordinators.sasha_coordinator import SashaCoordinatorConfig
from sage.testing.testcases import get_tests
from sage.testing.testcases import TEST_REGISTER
from sage.testing.testing_utils import current_save_dir
from sage.testing.testing_utils import get_base_device_state
from sage.testing.testing_utils import get_min_device_state
from sage.utils.common import CONSOLE
from sage.utils.llm_utils import ClaudeConfig
from sage.utils.llm_utils import GPTConfig
from sage.utils.llm_utils import TGIConfig


def merge_test_types(test_log: dict[str, Any]):
    """Merge testcases from different types"""
    test_register_names = {
        k: set([x.__name__ for x in v]) for k, v in TEST_REGISTER.items()
    }

    for case in test_log.keys():
        case_test_types = []

        for test_type, tests in test_register_names.items():
            if case in tests:
                case_test_types.append(test_type)
        test_log[case]["types"] = case_test_types


class CoordinatorType(Enum):
    """Coordinator type for config"""

    SAGE = SAGECoordinatorConfig
    SASHA = SashaCoordinatorConfig
    ZEROSHOT = OnePromptCoordinatorConfig


class LlmType(Enum):
    """LLM type for config"""

    GPT = GPTConfig
    CLAUDE = ClaudeConfig
    LEMUR = TGIConfig


@dataclass
class TestDemoConfig:
    trigger_server_url: str = f"http://{os.getenv('TRIGGER_SERVER_URL')}"
    trigger_servers: tuple[tuple] = (("condition", trigger_server_url),)
    coordinator_type: CoordinatorType = CoordinatorType.SAGE
    llm_type: LlmType = LlmType.GPT
    model_name: str = "gpt-4"
    wandb_tracing: bool = False
    logpath: str = "test"
    evaluator_llm = GPTConfig(model_name="gpt-4", temperature=0.0).instantiate()

    # Resume the run from a previous run folder
    resume_from: str = None
    # resume_from: str = "latest"
    # whether to skip successful testcases when resuming
    skip_passed: bool = True
    # whether to skip failed testcases when resuming
    skip_failed: bool = True
    # whether to include human interaction test cases
    include_human_interaction: bool = False
    # whether to include gmail and google calendar test cases
    enable_google: bool = False

    # test scenario : in or out of distribution
    test_scenario: str = "in-dist"

    def __post_init__(self):
        if self.llm_type.name == "LEMUR":
            self.llm_config = self.llm_type.value()
        else:
            self.llm_config = self.llm_type.value(model_name=self.model_name)

        coord_kwargs = {"llm_config": self.llm_config, "run_mode": "test"}

        if self.coordinator_type.name == "SAGE":
            coord_kwargs["enable_google"] = self.enable_google
        self.coordinator_config = self.coordinator_type.value(**coord_kwargs)

    def print_to_terminal(self):
        CONSOLE.rule("Test Config")
        CONSOLE.log(self)
        CONSOLE.rule("")
        CONSOLE.rule("Coordinator Config")
        CONSOLE.log(self.coordinator_config)
        CONSOLE.rule("")

    def save(self, logpath):
        config_yaml_path = Path(os.path.join(logpath, "test_config.yaml"))
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        config_yaml_path.write_text(yaml.dump(self), "utf8")

        coord_config_yaml_path = Path(os.path.join(logpath, "coord_config.yaml"))
        coord_config_yaml_path.write_text(yaml.dump(self.coordinator_config), "utf8")


def main(test_demo_config: TestDemoConfig):
    test_demo_config.print_to_terminal()

    save_dir = Path(os.getenv("SMARTHOME_ROOT")).joinpath(test_demo_config.logpath)

    if test_demo_config.wandb_tracing:
        os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
        os.environ["WANDB_PROJECT"] = "langchain-tracing"

    condition_server_url = None

    for name, url in test_demo_config.trigger_servers:
        if name == "condition":
            condition_server_url = url

    BaseConfig.global_config = GlobalConfig(
        condition_server_url=condition_server_url, 
        docmanager_cache_path=Path(os.getenv("SMARTHOME_ROOT")).joinpath(
            "external_api_docs/cached_test_docmanager.json"
        )
    )

    if test_demo_config.resume_from:
        if test_demo_config.resume_from == "latest":
            all_logs = [
                p for p in save_dir.rglob("*/*.json") if "initial" not in str(p)
            ]

            log_times = [datetime.fromisoformat(p.stem) for p in all_logs]
            resume_from = all_logs[np.argmax(log_times)]
        else:
            resume_from = test_demo_config.resume_from

        with open(resume_from, "r") as f:
            test_log = json.load(f)
        CONSOLE.print(f"[yellow]Test resumed from {resume_from}")
    else:
        test_log = {}

    os.makedirs(save_dir, exist_ok=True)
    now_str = str(datetime.now())
    save_detail_dir = save_dir.joinpath(now_str)
    save_path = save_detail_dir.joinpath(now_str + ".json")
    os.makedirs(save_detail_dir)
    current_save_dir[0] = save_detail_dir
    CONSOLE.log(f"[yellow]Saving logs in {save_detail_dir}")
    test_demo_config.save(save_detail_dir)

    if test_demo_config.test_scenario == "in-dist":
        test_cases = list(
            set(get_tests(list(TEST_REGISTER.keys()), combination="union"))
            - set(get_tests(["test_set"]))
        )
    else:
        test_cases = get_tests(["test_set"])

    if not test_demo_config.include_human_interaction:
        human_interaction_cases = get_tests(["human_interaction"])
        test_cases = list(set(test_cases) - set(human_interaction_cases))

    if not test_demo_config.enable_google:
        google_cases = get_tests(["google"])
        test_cases = list(set(test_cases) - set(google_cases))

    for case_func in test_cases:
        try:
            CONSOLE.print(f"Starting : {case_func}")
            case = case_func.__name__
            # Use reduced state for OnePromptCoordinator to avoid input with > tokens

            if isinstance(
                test_demo_config.coordinator_config, OnePromptCoordinatorConfig
            ):
                device_state = deepcopy(get_min_device_state())
            else:
                # SAGE or Sasha
                device_state = deepcopy(get_base_device_state())

            if case in test_log:
                result = test_log[case]["result"]

                if (result == "success") and test_demo_config.skip_passed:
                    CONSOLE.print("pass success")

                    continue

                if (result == "failure") and test_demo_config.skip_failed:
                    CONSOLE.print("pass failure")

                if "error" not in test_log[case]:
                    continue
                error_message = test_log[case]["error"]

                if not (
                    ("choices" in error_message)
                    or ("Client.generate()" in error_message)
                    or ("ChatAnthropic" in error_message)
                    or ("HTTPConnectionPool" in error_message)
                ):
                    continue

            start_time = time.time()

            case_func(device_state, test_demo_config)

            end_time = time.time() - start_time
            test_log[case] = {
                "case": case,
                "result": "success",
                "runtime": end_time,
            }
            CONSOLE.log(f"[green]\ncase {case} WIN  \U0001F603")
        except Exception as e:
            traceback.print_exc()
            test_log[case] = {
                "case": case,
                "result": "failure",
                "error": str(e),
            }
            CONSOLE.log(f"[red]\ncase {case} Fail \U0001F914")

        merge_test_types(test_log)
        with open(save_path, "w") as f:
            json.dump(test_log, f)

    merge_test_types(test_log)
    with open(save_path, "w") as f:
        json.dump(test_log, f)
    CONSOLE.print("DONE!")
    CONSOLE.log(
        "Success rate: ",
        len([t for t in test_log.values() if t["result"] == "success"]) / len(test_log),
    )


if __name__ == "__main__":
    main(tyro.cli(TestDemoConfig))
