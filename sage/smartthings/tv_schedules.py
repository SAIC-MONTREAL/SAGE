"""
Getting TV schedules
"""
import csv
import datetime
import os
from dataclasses import dataclass
from dataclasses import field
from typing import Type

import numpy as np

from sage.base import BaseToolConfig
from sage.base import SAGEBaseTool
from sage.smartthings.db import TvScheduleDb
from sage.utils.common import load_embedding_model
from sage.utils.common import parse_json

ROOT = os.getenv("SMARTHOME_ROOT", default=None)

if ROOT is None:
    raise ValueError("Env variable $SMARTHOME_ROOT is not set up.")


def add_embeddings(schedule):
    """
    Add embeddings to the schedule.

    Does things in place, so the original schedule is mutated.

    Args:
        schedule (list)

    Returns:
        schedule (list)
    """
    emb_function = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    texts = [
        x["channel_name"] + " - " + x["program_name"] + ": " + (x["program_desc"] or "")
        for x in schedule
    ]
    embed = emb_function.embed_documents(texts)

    for s, e in zip(schedule, embed):
        s["all-MiniLM-L6-v2"] = e

    return schedule


db = TvScheduleDb()


@dataclass
class QueryTvScheduleToolConfig(BaseToolConfig):
    """
    Config for QueryTvScheduleTool

    The source argument is included to allow the tool to search over multiple TV sources
    (e.g. cable TV and free Samsung TV programming). Currently, only a single source is supported
    (montreal-fibe-tv) but this can easily be modified.
    """

    _target: Type = field(default_factory=lambda: QueryTvScheduleTool)
    name: str = "tv_schedule_search"

    description: str = """
Search for programming currently playing on the TV.
Input a json string with keys:
source (str): id of the source [montreal-fibe-tv]
query (str)
"""
    # How many results to return
    top_k: int = 5
    # If true, add test programs to query corpus (i.e. pretend that the test programs are currently on)
    inject_test: bool = False
    # If true, ignore tv schedule DB and only incude test programs
    injected_only: bool = True


class QueryTvScheduleTool(SAGEBaseTool):
    top_k: int = None
    inject_test: bool = None
    injected_only: bool = None

    def _inject(self, on_now: list) -> list:
        """
        Inject information about what is playing on the fake TV right now.

        The aim of this is to support testing by making sure that the show we want to play
        is always on.
        """

        if not self.inject_test:
            return on_now

        test_tv_path = f"{ROOT}/sage/testing/tv_guide.csv"

        with open(test_tv_path, "r") as test_csv:
            reader = csv.DictReader(test_csv)
            test_tv_guide_lst = []
            for dct in reader:
                test_tv_guide_lst.append(dct)

        add_embeddings(test_tv_guide_lst)

        # Make channel_number-indexed dictionary for use
        # when replacing channels in the `on_now` object.
        #   i.e. { 0:{chan0_dict},  1:{chan1_dict}, ...}
        # Also set program time to current time.
        test_tv_guide = {}
        for dct in test_tv_guide_lst:
            dct["start_ts"] = (
                datetime.datetime.utcnow() - datetime.timedelta(minutes=15)
            ).strftime("00:%H:%M")
            dct["end_ts"] = (
                datetime.datetime.utcnow() + datetime.timedelta(minutes=45)
            ).strftime("00:%H:%M")

            # index by channel number
            test_tv_guide[int(dct["channel_number"])] = dct

        # replace any channels with same channel_number as test tv channels
        for i, chan in enumerate(on_now):
            if len(test_tv_guide) == 0:
                # all test tv channels have been injected
                break
            elif int(chan[0]["channel_number"]) in test_tv_guide.keys():
                tmp = test_tv_guide.pop(int(chan[0]["channel_number"]))
                on_now[i] = [tmp]

        # add any channels that didn't replace an existing one
        for test_chan in test_tv_guide:
            on_now.append([test_tv_guide[test_chan]])

        return on_now

    def setup(self, config: QueryTvScheduleToolConfig) -> None:
        self.top_k = config.top_k
        self.injected_only = config.injected_only
        if self.injected_only:
            self.inject_test = True
        else:
            self.inject_test = config.inject_test

    def _run(
        self,
        command: str,
    ) -> str:
        emb = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")

        parsed_command = parse_json(command)
        if parsed_command is None:
            return "Invalid input format. Input a json string with keys: source (str) and query (str)."

        source = parsed_command["source"]
        query = parsed_command["query"]

        if self.injected_only:
            on_now = []
        else:
            # update this logic if adding new sources
            if "fibe" in source.lower():
                provider_string = "montreal-fibe-tv"
            else:
                return "value of query argument must be montreal-fibe-tv"
            on_now = db.whats_on(provider_string)

        on_now = self._inject(on_now)
        query_embed = np.array(emb.embed_query(query))[None, :]
        on_now_embed = np.array([x[0]["all-MiniLM-L6-v2"] for x in on_now])

        sim = (query_embed @ on_now_embed.T).squeeze()
        argbest = np.argsort(-sim)[: self.top_k]
        out = ["Here are some relevant TV programs that are on now:\n"]
        for idx in argbest:
            out.append(
                "Channel number: %s. Channel name: %s. Program name: %s. Program description: %s \n"
                % (
                    on_now[idx][0]["channel_number"],
                    on_now[idx][0]["channel_name"],
                    on_now[idx][0]["program_name"],
                    on_now[idx][0]["program_desc"],
                )
            )
        return "\n".join(out)

    async def _arun(self, *args, **kwargs):
        return NotImplementedError
