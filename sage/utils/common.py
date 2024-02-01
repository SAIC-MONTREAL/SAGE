"""
Common util functions
"""
import json
import os
import time
from functools import lru_cache
from inspect import currentframe
from inspect import getsource
from typing import Any
from typing import List

import numpy as np
import yaml
from box import Box
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.output_parsers.json import parse_json_markdown
from rich.console import Console


def check_env_vars(hf_setup: bool = False):
    """Check if the environment variables are set"""

    root_path = os.getenv("SMARTHOME_ROOT", default=None)

    if root_path is None:
        raise ValueError("Env variable $SMARTHOME_ROOT is not set up.")

    if hf_setup:
        HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", default=None)

        # Max retries fix:
        # https://stackoverflow.com/questions/75110981/sslerror-httpsconnectionpoolhost-huggingface-co-port-443-max-retries-exce
        os.environ["CURL_CA_BUNDLE"] = ""

        if HUGGINGFACEHUB_API_TOKEN is None:
            raise ValueError("Please provide HF API token")


def parse_json(json_string: str) -> Any:
    """
    Validates if a JSON string is formatted correctly.
    Also verifies if a string in markdown format can be parsed as JSON.
    It is helpful because some LLMs will output correct JSON put in markdown format
    """

    try:
        return json.loads(json_string)

    except json.JSONDecodeError:
        try:
            return parse_json_markdown(json_string)

        except Exception:
            return None


def read_config(config_path: str) -> Box:
    """Loads config from yaml file"""
    with open(config_path, "r", encoding="utf8") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))

    return cfg


def save_config(path: str, object_to_save: Any) -> None:
    """Save an object to disk as a yaml file"""
    assert os.path.splitext(path)[1] == ".yaml"
    print(f"Writing metadata to {path}")

    with open(path, "w") as fp:
        yaml.dump(object_to_save, fp)


def read_json(filename: str) -> dict[str, Any]:
    """Reads a json file"""

    return json.load(open(filename, "r"))


def findall(long_string: str, short_string: str) -> List[int]:
    """
    Find indices of all occurrences of short_string in long_string.
    """
    out = []
    idx_start = 0

    while True:
        idx = long_string.find(short_string)

        if idx == -1:
            return out
        out.append(idx + idx_start)
        long_string = long_string[idx + len(short_string) :]
        idx_start += idx + len(short_string)


def function2string(function_handle: str, code_define: str) -> str:
    """convert a function handle to string and append it to code_define program string"""
    code_define += "\n"
    code_define += getsource(function_handle)

    return code_define


@lru_cache(maxsize=None)
def load_embedding_model(model_name: str):
    """Builds an embedding model and cache it"""

    return HuggingFaceEmbeddings(model_name=model_name)


class Timeliner:
    """
    Nifty little utility to time how long stuff takes.
    Usage:
    from wat.utils import tl
    tl.timeline()
    do_stuff(args)
    tl.timeline()
    every time you call tl.timeline() it prints the line number and the
    amount of time that has passed since the last call to tl.timeline().
    """

    def __init__(self):
        self.t = time.time()
        self.last_nan_count = None

    def timeline(self, nan_count=None):
        cf = currentframe()
        tnew = time.time()
        print(
            "pid",
            os.getpid(),
            cf.f_back.f_code.co_filename,
            cf.f_back.f_lineno,
            tnew - self.t,
        )

        if nan_count is not None:
            n_nans = np.isnan(nan_count).sum()
            head = ""
            tail = ""
            # this will cause the line to be printed in red, cool huh?

            if self.last_nan_count is not None and self.last_nan_count != n_nans:
                head = "\033[91m"
                tail = "\033[0m"
            self.last_nan_count = n_nans
            print(head, "    nan count", np.isnan(nan_count).sum(), tail)

        self.t = tnew


tl = Timeliner()
CONSOLE = Console(width=120)
