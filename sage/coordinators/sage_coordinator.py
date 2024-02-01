""" Sage Coordinator """
import os
import pickle
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from typing import Any
from typing import Type

from langchain.agents import initialize_agent

from sage.base import BaseToolConfig
from sage.coordinators.base import AgentConfig
from sage.coordinators.base import BaseCoordinator
from sage.coordinators.base import CoordinatorConfig
from sage.human_interaction.tools import HumanInteractionToolConfig
from sage.misc_tools.weather_tool import WeatherToolConfig
from sage.retrieval.memory_bank import MemoryBank
from sage.retrieval.tools import UserProfileToolConfig
from sage.smartthings.persistent_command_tools import ConditionCheckerToolConfig
from sage.smartthings.persistent_command_tools import NotifyOnConditionToolConfig
from sage.smartthings.smartthings_tool import SmartThingsToolConfig
from sage.smartthings.tv_schedules import QueryTvScheduleToolConfig
from sage.utils.llm_utils import TGIConfig
from sage.utils.logging_utils import initialize_tool_names


@dataclass
class SAGECoordinatorConfig(CoordinatorConfig):
    """SAGE coordinator config"""

    _target: Type = field(default_factory=lambda: SAGECoordinator)

    name: str = "SAGE"
    agent_type: str = "zero-shot-react-description"
    agent_config: AgentConfig = AgentConfig()
    memory_path: str = os.path.join(
        os.getenv("SMARTHOME_ROOT"), "data/memory_data/large_memory_bank.json"
    )
    # Bool to activate the memory updating
    enable_memory_updating: bool = False

    # Bool to activate human interaction
    enable_human_interaction: bool = False

    # Bool to activate google tool
    enable_google: bool = False

    # Bool to use the same llm config for all the tools
    single_llm_config: bool = True

    # The tools config
    tool_configs: tuple[BaseToolConfig, ...] = (
        UserProfileToolConfig(),
        SmartThingsToolConfig(),
        QueryTvScheduleToolConfig(),
        ConditionCheckerToolConfig(),
        NotifyOnConditionToolConfig(),
        WeatherToolConfig(),
        HumanInteractionToolConfig(),
    )

    # Save a snapshot of the memory of N interactions
    snapshot_frequency: int = 1

    def __post_init__(self):
        super().__post_init__()
        
        if self.enable_google:
            from sage.misc_tools.google_suite import GoogleToolConfig

            self.tool_configs = self.tool_configs + (GoogleToolConfig(),)

        if self.single_llm_config:
            self.override_llm_config(tool_configs=self.tool_configs)

    def override_llm_config(self, tool_configs: tuple[BaseToolConfig]) -> None:
        """Overrides the LLM config for the tools based on the coordinator config"""

        for config in tool_configs:
            if len(config.tool_configs) > 0:
                self.override_llm_config(config.tool_configs)

            if hasattr(config, "llm_config"):
                # override config using the coordinator config
                config.llm_config = self.llm_config


class SAGECoordinator(BaseCoordinator):
    """SAGE coordinator instantiates agents, llms and tools"""

    def __init__(self, config: SAGECoordinatorConfig):
        super().__init__(config)

        self.tooldict = {}
        self.memory = MemoryBank()
        memory_save_dir = os.path.join(config.global_config.logpath, "memory_snapshots")
        os.makedirs(memory_save_dir, exist_ok=True)

        if os.path.isfile(os.path.join(memory_save_dir, "initial_snapshot.json")):
            self.memory.read_from_json(
                os.path.join(memory_save_dir, "initial_snapshot.json")
            )
        else:
            self.memory.read_from_json(config.memory_path)

            self.memory.save_snapshot(
                os.path.join(memory_save_dir, "initial_snapshot.json")
            )

        if isinstance(config.llm_config, TGIConfig):
            config.llm_config = TGIConfig(stop_sequences=["Human", "Question"])

        # setup tools
        self._build_tools()

        # save tool descriptions in logs (used in visualization)
        tool_file = os.path.join(config.global_config.logpath, "tools.pickle")

        if not os.path.exists(tool_file):
            with open(tool_file, "wb") as fp:  # Pickling
                pickle.dump(initialize_tool_names(self.tooldict), fp)

        # setup agent
        toollist = [tool for _, tool in self.tooldict.items()]
        self.agent = initialize_agent(
            toollist,
            self.llm,
            agent=config.agent_type,
            agent_kwargs=asdict(config.agent_config),
            handle_parsing_errors=True,
        )

        self.request_idx = 0

    def _build_tools(self) -> None:
        """Add tools to the agent"""

        for tool_config in self.config.tool_configs:
            if (
                not self.config.enable_human_interaction
                and tool_config.name == "human_interaction_tool"
            ):
                continue

            if self.config.enable_memory_updating or tool_config.name in [
                "human_interaction_tool",
                "user_preference_tool",
            ]:
                tool = tool_config.instantiate(memory=self.memory)
            else:
                tool = tool_config.instantiate()

            self.tooldict[tool.name] = tool

    def update_tools(self, kwargs: dict[str, Any]) -> None:
        """
        Update the path from which the tool will read the json files for device
        states and global states
        """

        for tool_name, tool in self.tooldict.items():
            if tool_name in kwargs:
                tool.update(kwargs[tool_name])

    def update_memory(self, user_name: str, command: str) -> None:
        """
        Update the user memory.
        """
        self.memory.add_query(user_name, command, str(date.today()))
        # create new memory tool because the memory is updated

        for tool_config in self.config.tool_configs:
            if tool_config.name == "user_profile_tool":
                self.tooldict["user_profile_tool"] = tool_config.instantiate(
                    memory=self.memory
                )

                break

        # save a snapshot after snapshot_frequency interactions
        self.request_idx += 1

        if self.request_idx % self.config.snapshot_frequency == 0:
            self.memory.save_snapshot(
                os.path.join(self.config.output_dir, "memory_snapshots")
            )

    def execute(self, command: str) -> str:
        """
        Runs the agent with the provided command
        """

        response = self.agent(command, callbacks=self.callbacks)

        if self.config.enable_memory_updating:
            user_name, query = command.split(":")
            user_name = user_name.lower().strip()
            self.update_memory(user_name, query)

        return response
