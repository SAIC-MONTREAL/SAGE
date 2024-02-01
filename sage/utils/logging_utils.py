"""Callback Handler that writes to a file."""
import os
from typing import Any
from typing import cast
from typing import Dict
from typing import Optional
from typing import TextIO

from langchain.callbacks import FileCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction
from langchain.schema import AgentFinish
from langchain.utils.input import print_text

from sage.base import SAGEBaseTool


def get_callback_handlers(
    logpath: str, logname: str = "experiment.log", viz_logname: str = "viz.log"
) -> list[BaseCallbackHandler]:
    """This function creates the 2 types of log handlers used in our demo, verbose and for graphics"""
    callback_handler = FileCallbackHandler(os.path.join(logpath, logname))
    callback_handler_viz = LogStatesActionCallbackHandler(
        os.path.join(logpath, viz_logname)
    )

    return [callback_handler, callback_handler_viz]


def find_all_substrings(string: str, substring: str) -> list[int]:
    """Find the indices of all substrings within a large string"""

    return [i for i in range(len(string)) if string.startswith(substring, i)]


def first_larger_term(list1: list[str], number: int) -> int:
    """
    Takes a list of numbers and a number to compare against and
    returns the first term in list1 that is larger than number, or None if no such term exists.
    """

    for term in list1:
        if term > number:
            return term
    # No term in list1 was larger than number.

    return None


def extract_texts(text: str, start_string: str, end_string: str) -> list[int]:
    """ "Extract all instances of text that occur between the string identifiers start_string and end_string"""
    start_indices = find_all_substrings(text, start_string)
    end_indices = find_all_substrings(text, end_string)
    # fhogan hack: end indices returns numbers that aren't associated with start terms sometimes.
    # Filter such as to remove outliers
    end_indices = [first_larger_term(end_indices, term) for term in start_indices]

    text_list = []

    for counter, final_index in enumerate(start_indices):
        start_index = start_indices[counter]
        end_index = end_indices[counter]

        if text[end_index - 1 : end_index] == "\n":
            end_index = end_index - 1
        text_list.append(text[start_index + len(start_string) : end_index])

    return text_list


def initialize_tool_names(tooldict: dict[str, SAGEBaseTool]) -> list[str]:
    """Get list of available tools, at every hierchy level of the coordinator agent"""
    tools_list = []
    tools_list.append([*tooldict.keys()])

    for tool in tooldict.keys():
        if len(tooldict[tool].tools) > 0:
            tools_list.append(
                [tooldict[tool].tools[i].name for i in range(len(tooldict[tool].tools))]
            )

    return tools_list


class LogStatesActionCallbackHandler(BaseCallbackHandler):
    """Custom Callback Handler that writes to a file."""

    def __init__(
        self, filename: str, mode: str = "a", color: Optional[str] = None
    ) -> None:
        """Initialize callback handler."""
        self.file = cast(TextIO, open(filename, mode, encoding="utf-8"))
        self.color = BaseCallbackHandler

    def __del__(self) -> None:
        """Destructor to cleanup when done."""
        self.file.close()

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        print_text(
            f"[CHAIN START] Entering new {class_name} chain...",
            end="\n",
            file=self.file,
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        print_text("[CHAIN END] Finished chain.", end="\n", file=self.file)

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        print_text(action.log, color=color or self.color, end="\n", file=self.file)

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        print_text(finish.log, color=color or self.color, end="\n", file=self.file)
