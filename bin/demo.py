"""
Main demo script

Examples of running the demo:

Option 1 (preloaded commands):
<add the commands you want to run to preloaded_commands>
python demo.py

Option 2 (manual triggers):
python demo.py
<in another terminal>
curl -d '{"user": "Amal", "command": "is the dishwasher running?"} -X POST http://<TRIGGER_SERVER_URL>/trigger_manually
"""
import multiprocessing as mp
import os

import time
from dataclasses import dataclass

import langchain
import requests
import tyro

from sage.base import BaseConfig
from sage.base import GlobalConfig
from sage.coordinators.sage_coordinator import SAGECoordinatorConfig
from sage.utils.common import check_env_vars
from sage.utils.trigger_server import AllServerRunner


@dataclass
class DemoConfig:
    """
    Config for demo setup.
    """

    use_ice: bool = False
    use_treeviz: bool = False
    # should be user, command tuples
    preloaded_commands: tuple[tuple] = (
        (
            "Amal",
            "Turn on the TV.",
        ),
    )
    trigger_server_url: str = f"http://{os.getenv('TRIGGER_SERVER_URL')}"
    trigger_servers: tuple[tuple] = (("condition", trigger_server_url),)
    verbose: bool = True
    wandb_tracing: bool = False


if __name__ == "__main__":
    check_env_vars()

    demo_config = DemoConfig()

    server_runner = AllServerRunner()
    server_runner.run()
    print("ran server")

    # initialize weights and biases tracing

    if demo_config.wandb_tracing:
        os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
        os.environ["WANDB_PROJECT"] = "langchain-tracing"

    # hacky way to communicate condition server url to the NotifyOnConditionTool
    condition_server_url = None

    for name, url in demo_config.trigger_servers:
        if name == "condition":
            condition_server_url = url

    BaseConfig.global_config = GlobalConfig(condition_server_url=condition_server_url)
    langchain.verbose = demo_config.verbose

    config = tyro.cli(SAGECoordinatorConfig)
    coordinator = config.instantiate()

    if demo_config.use_treeviz:
        from utils.viz_utils import visualize_agent

        ctx = mp.get_context("spawn")
        viz_process = ctx.Process(
            target=visualize_agent,
            args=(
                coordinator.config.global_config.logpath.split("/")[-1],
                5,
            ),
        )

    if demo_config.use_ice:
        from langchain_visualizer import visualize

    # slightly hacky way to maintain counter over multiple function calls
    next_preloaded_command = [0]

    def poll_triggers(demo_config):
        if len(demo_config.preloaded_commands) > next_preloaded_command[0]:
            out = demo_config.preloaded_commands[next_preloaded_command[0]]
            next_preloaded_command[0] += 1

            return out

        while True:
            for _, url in demo_config.trigger_servers:
                trigger = requests.get(url + "/check_triggers").json()
                print("got trigger from %s" % url, trigger)

                if trigger:
                    return trigger["user"], trigger["command"]
            time.sleep(1)

    while True:
        user, command = poll_triggers(demo_config)

        user_command = f"{user} : {command}"

        if demo_config.use_ice:

            async def run_agent_demo():
                return coordinator.execute(user_command)

            visualize(run_agent_demo)
        elif demo_config.use_treeviz:
            viz_process.start()
            coordinator.execute(user_command)
            viz_process.join()
        else:
            coordinator.execute(user_command)
