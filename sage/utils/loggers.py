"""
Logging functions

"""

from typing import Dict, Any
import os
import pandas as pd


class InferenceLogger:
    """The path of the output directory"""

    def __init__(self, output_dir: str) -> None:

        self.output_dir = output_dir

        # Create output dirs
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "responses"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)

        self.scores = []
        self.responses = []
        self.retrieved_memories = []
        self.format_issues = []
        self.commands = []
        self.case_ids = []
        self.references = []

    def add(
        self,
        case_id: str,
        llm_response: Dict[str, Any],
        testcase: Dict[str, Any],
        extras: Dict[str, Any],
    ) -> None:
        """update with new intermediate result"""
        self.case_ids.append(case_id)
        self.commands.append(testcase["command"])
        self.references.append(testcase["outcome"])
        self.responses.append(llm_response)
        self.format_issues.append(extras["matching_failed"])

        self.scores.append(extras["score"])

        if "source_documents" in extras.keys():
            self.retrieved_memories.append(
                [
                    doc.page_content.replace("  ", "")
                    for doc in extras["source_documents"]
                ]
            )

    def save_response(self, response: str, case_id: str) -> None:
        """save raw LLM response"""
        with open(f"{self.output_dir}/responses/{case_id}.txt", "w") as f:
            f.write(response)

    def save_results(self) -> None:
        """save final results in a csv file"""
        results = {
            "ID": self.case_ids,
            "Command": self.commands,
            "Pred": self.responses,
            "Reference": self.references,
            "RougeL": self.scores,
            "FormatFail": self.format_issues,
        }

        if len(self.retrieved_memories) > 0:
            results["Retrieved"] = self.retrieved_memories

        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv(os.path.join(self.output_dir, "scores.csv"), index=False)
