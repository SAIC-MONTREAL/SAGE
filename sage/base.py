"""All our base classes"""
from pathlib import Path
import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type

from langchain.tools import BaseTool


class SAGEBaseTool(BaseTool):
    # Tools can be hierarchical
    tools: List[BaseTool] = None

    def setup(self, config: Dict[str, Any]) -> None:
        """Tool-specific setup"""

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class GlobalConfig:
    """
    Config that is not really specific to any tool, but might be used by multiple tools.
    This should pretty much be meta-information about the test run, not actually anything that
    affects the performance of the tools.
    """

    # path where logs are stored
    logpath: str = None

    # url of server to use for conditions
    condition_server_url: str = None

    # test_id, only set to non-null if running testing
    test_id: str = None

    smartthings_token: str = os.getenv("SMARTTHINGS_API_TOKEN")

    docmanager_cache_path: Path = Path(os.getenv("SMARTHOME_ROOT")).joinpath(
    "external_api_docs/cached_real_docmanager.json"
)


@dataclass
class BaseConfig:
    """Config class for instantiating the class specified in the _target attribute."""

    global_config: ClassVar[GlobalConfig]

    _target: Type

    def instantiate(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""

        return self._target(self, **kwargs)

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]

        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["

                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")

        return "\n    ".join(lines)


@dataclass
class BaseToolConfig(BaseConfig):
    """Config class for instantiating tools"""

    # NOTE : All the configurable fields of your tool (e.g., description), should be defined in the
    # config and not the tool class
    name: str = None
    description: str = None

    tool_configs: tuple[BaseConfig, ...] = tuple()
    tools: list[SAGEBaseTool] = field(default_factory=lambda: [])

    def instantiate(self, **kwargs):
        if self.tool_configs:
            self.tools = [
                tool_config.instantiate() for tool_config in self.tool_configs
            ]

        obj = self._target(
            name=self.name, description=self.description, tools=self.tools
        )
        obj.setup(self, **kwargs)

        return obj
