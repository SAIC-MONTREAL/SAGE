from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type, Dict, List
import json

import pandas as pd

from testing.testing_utils import get_base_device_state, current_save_dir
from testing.testcases import get_tests, TEST_REGISTER, get_test_challenges

from base import GlobalConfig, BaseConfig

test_cases = set(get_tests(list(TEST_REGISTER.keys()), combination="union"))

test_category_lookup = {
    "device_resolution": "DR",
    "personalization": "P",
    "persistence": "T",
    "intent_resolution": "IR",
    "command_chaining": "CR",
    "simple": "DC",
    "test_set": "TS",
}


class NullCoordinator:
    def __init__(self, config):
        pass

    def execute(self, command):
        raise ValueError(command)


@dataclass
class NullCoordinatorConfig(BaseConfig):
    """Config to run a demo with OnePrompt baseline."""

    _target: Type = field(default_factory=lambda: NullCoordinator)
    name: str = "null"


@dataclass
class TestDemoConfig:
    coordinator_config: NullCoordinatorConfig
    trigger_servers: tuple[tuple] = (("condition", "http://0.0.0.0:5797"),)


test_demo_config = TestDemoConfig(NullCoordinatorConfig())

current_save_dir[0] = Path("logs/blah")
BaseConfig.global_config = GlobalConfig(
    condition_server_url=test_demo_config.trigger_servers[0][1]
)


def testcase_table():
    ids = []
    texts = []
    types = []
    for case_func in test_cases:
        device_state = deepcopy(get_base_device_state())
        try:
            case_func(device_state, test_demo_config)
        except ValueError as e:
            text = str(e)
            texts.append(text)
            case_id = case_func.__name__
            case_id = case_id.replace("_", "\_")
            ids.append(case_id)
            types.append(
                ", ".join(
                    [test_category_lookup[x] for x in get_test_challenges(case_func)]
                )
            )

    table_header = """
    \\onecolumn
    \\tablehead{\\hline
    \\textbf{ID} & \\textbf{Query} & \\textbf{Categories} \\\\
    \\hline
    }
    \\tabletail{\\hline \\multicolumn{3}{|r|}{Continued on the next page} \\\\ \\hline}


    \\begin{xtabular}{|p{8cm}|p{5cm}|p{2cm}|}
    \\hline
    """

    table_content = "\n".join(
        ["%s & %s & %s \\\\ \n \hline" % tup for tup in zip(ids, texts, types)]
    )

    table_footer = """
    \\end{xtabular}
    \\label{table:testcases}
    \\twocolumn
    """
    table = table_header + table_content + table_footer
    with open("logs/cases_table.tex", "w") as f:
        f.write(table)

    # Now dump all results


def read_all_results(base_path=Path("/Users/d.rivkin/work/smarthome-llms/logs/")):
    rows = []
    for p in base_path.rglob("*.json"):
        if "initial_snapshot" in str(p):
            continue
        if "failure_analysis" in str(p):
            continue
        print(p)
        rows.append(
            {
                "path": str(p),
                "time": p.stem,
                "trial": p.parent.parent.stem,
                "llm": p.parent.parent.parent.stem,
                "method": p.parent.parent.parent.parent.stem,
                "test_type": p.parent.parent.parent.parent.parent.stem,
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("time", ascending=False)
    df = df.drop_duplicates(["test_type", "method", "llm", "trial"], keep="first")
    run_results = []
    for _, run in df.iterrows():
        with open(run["path"], "r") as f:
            results = json.load(f)
        for case_id, case_result in results.items():
            run_results.append(
                {
                    "method": run["method"],
                    "trial": run["trial"],
                    "llm": run["llm"],
                    "test_id": case_id,
                    "result": case_result["result"],
                    # 'runtime': case_result['runtime']
                }
            )
    df_out = pd.DataFrame(run_results)
    df_out = df_out.sort_values(["method", "llm", "test_id", "trial"])
    return df_out


df = read_all_results()

table_header = """
\\onecolumn
\\tablehead{\\hline
\\textbf{Method} & \\textbf{LLM} & \\textbf{Trial} & \\textbf{ID} & \\textbf{Result} \\\\
\\hline
}
\\tabletail{\\hline \\multicolumn{5}{|r|}{Continued on the next page} \\\\ \\hline}

\\begin{xtabular}{|p{2cm}|p{3cm}|p{2cm}|p{7cm}|p{2cm}|}
\\hline
"""


table_footer = """
\\end{xtabular}
\\label{table:testcases}
\\twocolumn
"""

table_lines = []
for _, row in df.iterrows():
    test_id = row["test_id"].replace("_", "\_")
    trial = row["trial"].strip("trial_")
    llm = row["llm"]
    method = {"SAGE": "SAGE", "SASHA": "Sasha", "ZeroShot": "One Prompt"}[row["method"]]
    line = "%s & %s & %s & %s & %s \\\\ \n \hline" % (
        method,
        llm,
        trial,
        test_id,
        row["result"],
    )
    table_lines.append(line)

table_content = "\n".join(table_lines)

table = table_header + table_content + table_footer
with open("logs/success_table.tex", "w") as f:
    f.write(table)


# columns are test_id, method, model, run, success, annotations
