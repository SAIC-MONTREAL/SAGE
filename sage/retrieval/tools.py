"""Create a memory retrieval tool for agents"""
import os
from typing import Dict, Any, Type
from dataclasses import dataclass, field
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
from langchain import LLMChain

from sage.utils.llm_utils import LLMConfig, TGIConfig
from sage.retrieval.templates import tool_template
from sage.retrieval.memory_bank import MemoryBank
from sage.base import SAGEBaseTool, BaseToolConfig

from sage.utils.common import parse_json


@dataclass
class UserProfileToolConfig(BaseToolConfig):

    _target: Type = field(default_factory=lambda: UserProfileTool)

    name: str = "user_preference_tool"
    description: str = """
Use this to learn about the user preferences and retrieve past interactions with the user.
Use this tool before addressing subjective or ambiguous user commands that require personalized information about a user.
This tool is not capable of asking the user anything.
The query should be clear, precise and formatted as a question.
The query should specify which type of preferences you are looking for.
Input should be a json string with 2 keys: query and user_name.
"""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vectordb: str = "chroma"
    memory_path: str = f"{os.getenv('SMARTHOME_ROOT')}/memory_data/memory_bank.json"
    loader_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"jq_schema": ".instruction"}
    )
    top_k: int = 5

    llm_config: LLMConfig = None


class UserProfileTool(SAGEBaseTool):
    """
    Defines a user profile tool to check user preferences.
    """

    top_k: int = 5
    llm: BaseLLM = None
    memory: MemoryBank = None

    def setup(self, config: UserProfileToolConfig, memory=None) -> None:
        """Setup the memory retrieval chain"""
        self.top_k = config.top_k

        if memory is None:
            self.memory = MemoryBank()
            self.memory.read_from_json(config.memory_path)
            self.memory.create_indexes(
                config.vectordb, config.embedding_model, load=True
            )
        else:
            self.memory = memory
            self.memory.create_indexes(
                config.vectordb, config.embedding_model, load=False
            )

        if isinstance(config.llm_config, TGIConfig):
            config.llm_config = TGIConfig(stop_sequences=["Question"])
        self.llm = config.llm_config.instantiate()

    def _run(self, text: str) -> str:
        attr = parse_json(text)

        if attr is None:
            return "The input does not follow the required format. The input should be a json string with 2 keys: query and user_name. Can you try again?"

        attr["user_name"] = attr["user_name"].lower()

        if attr["user_name"] not in self.memory.history.keys():
            return "No such user: %s. Known users are: %s" % (
                attr["user_name"],
                ",".join(list(self.memory.history.keys())),
            )

        memories = self.memory.search(**attr, top_k=self.top_k)
        preferences = self.memory.history[attr["user_name"]]["profile"]

        prompt = PromptTemplate.from_template(tool_template)

        inputs = {
            "preferences": preferences,
            "context": memories,
            "username": attr["user_name"],
            "question": attr["query"],
        }
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        response = llm_chain.predict(**inputs)

        return response


if __name__ == "__main__":

    import langchain
    import tyro
    from coordinators.coordinator import CoordinatorConfig

    langchain.verbose = True

    config = tyro.cli(CoordinatorConfig)
    coordinator = config.instantiate()

    coordinator.execute("Amal: What is my favorite food?")
