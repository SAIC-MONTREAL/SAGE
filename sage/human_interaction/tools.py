from typing import Dict, Any, Type
from dataclasses import dataclass, field
from datetime import date
from sage.retrieval.memory_bank import MemoryBank
from sage.base import SAGEBaseTool, BaseToolConfig


@dataclass
class HumanInteractionToolConfig(BaseToolConfig):
    _target: Type = field(default_factory=lambda: HumanInteractionTool)
    name: str = "human_interaction_tool"
    description: str = """Use this tool to communicate with the user. This
    can be to interact with the user to ask for more information on a topic or
    clarification on a previously requested command. Pass the query in a json with
    key "query".
    """


class HumanInteractionTool(SAGEBaseTool):
    """
    This tool is provided to the Coordinator to help it communicate with the
    user. This can be in numerous usecases like getting feedback, asking for more
    information or just having a conversation. The interaction can be done
    both via text or speech depending on the configuration setup.
    """

    memory: MemoryBank = None

    def setup(self, config: HumanInteractionToolConfig, memory=None):
        """
        Setup the AudioReader class
        """
        self.memory = memory

    def _run(self, dummy_string):
        """
        Returns the username and command utterance.
        """

        text_input = input("\nType (<username> : <command>) >> ")
        user_name, command = text_input.split(":")

        if self.memory:
            self.memory.add_query(
                user_name.strip().lower(), command.strip(), str(date.today())
            )
        return command.strip(), user_name.strip().lower()


if __name__ == "__main__":
    interaction_config = HumanInteractionToolConfig()
    interaction_tool = interaction_config.instantiate()
