"""
Base coordinator and config
"""
import os
import secrets
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Optional

import yaml

from sage.base import BaseConfig
from sage.coordinators.prompts import ACTIVE_REACT_COORDINATOR_PREFIX
from sage.coordinators.prompts import ACTIVE_REACT_COORDINATOR_SUFFIX
from sage.utils.common import CONSOLE
from sage.utils.llm_utils import GPTConfig
from sage.utils.llm_utils import LLMConfig
from sage.utils.llm_utils import TGIConfig
from sage.utils.logging_utils import get_callback_handlers


@dataclass(frozen=True)
class AgentConfig:
    """SAGE Agent configuration"""

    max_iters: str = 5
    prompt: str = None
    verbose: Optional[bool] = True
    prefix: str = ACTIVE_REACT_COORDINATOR_PREFIX
    suffix: str = ACTIVE_REACT_COORDINATOR_SUFFIX


@dataclass
class CoordinatorConfig(BaseConfig):
    """Coordinator config"""

    name: str = ""

    # The LLM configuration
    llm_config: LLMConfig = field(default_factory=lambda: GPTConfig())

    # Verbosity
    verbose: bool = True

    # In which mode the coordinator should run, e.g., test, dev, prod
    run_mode: str = "test"

    # Load existing config
    load_config: Optional[Path] = None

    def print_to_terminal(self) -> None:
        """Helper to pretty print config to terminal"""
        CONSOLE.rule("Config")
        CONSOLE.print(self)
        CONSOLE.rule("")

    def save_config(self) -> None:
        """Save config to output_dir directory"""

        config_yaml_path = Path(os.path.join(self.global_config.logpath, "config.yaml"))
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        config_yaml_path.write_text(yaml.dump(self), "utf8")

    def __post_init__(self):

        if self.verbose:
            import langchain

            langchain.verbose = True


class BaseCoordinator:
    """
    Base class for coordinators. Coordinators are agents that
    use tools or other agents as tools to address user commands.
    """

    def __init__(self, config: CoordinatorConfig):

        self.config = config

        if config.load_config:
            CONSOLE.log(f"Loading config from {config.load_config}")
            config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)

        if self.config.verbose and self.config.run_mode != "test":
            config.print_to_terminal()

        if self.config.run_mode != "test":
            config.save_config()

        if config.global_config.logpath is None:
            config.global_config.logpath = os.path.join(
                os.getenv("SMARTHOME_ROOT"), "logs/demo_runs", secrets.token_hex(16)
            )
        os.makedirs(config.global_config.logpath, exist_ok=True)

        self.callbacks = get_callback_handlers(config.global_config.logpath)

        # setup llm

        if isinstance(config.llm_config, TGIConfig):
            config.llm_config = TGIConfig(stop_sequences=["<FINISHED>"])

        self.llm = config.llm_config.instantiate()

    def execute(self, command: str, **kwargs: dict[str, Any]) -> str:

        raise NotImplementedError
