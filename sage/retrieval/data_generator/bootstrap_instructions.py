"""
Generate alexa-like commands.
Inspired from the self-instruct codebase

This code generates new instructions from a small set of seed human-written instructions in a bootstrapping fashion. The generated machine instructions are one-shot interaction with a smart home assistance.

Examples of bootstapped instructions:
    Play jazz music
    Turn on the volume
    Turn off the living room lights

"""
import os
import ast
import json
from typing import List, Dict, Any, Tuple, Optional
import re
import random
import datetime
from multiprocessing import Pool
from functools import partial

import tqdm
from argparse import ArgumentParser, Namespace
import glob
import pandas as pd

from rouge_score import rouge_scorer

from sage.utils.llm_utils import make_chatgpt_request
from sage.retrieval.data_generator.outputs import DatasetWriter

GENERIC_PROMPT = "Come up with a series of commands for a smart home assistant. Try to specify in parenthesis the device related to the command:\n"

USER_PROFILE_PROMPT = """I want you to act like a person who talks to their home assistant naturally. Your preferences are :\n
{preferences}
Come up with a series of general commands you might say to your smart home assistant.\n
"""


PROMPT_TEMPLATES = {"generic": GENERIC_PROMPT, "user_profile": USER_PROFILE_PROMPT}


def encode_prompts(
    prompt_instructions: List[str], prompt: str = GENERIC_PROMPT, inputs=None
) -> str:
    """Encode multiple prompt instructions into a single string."""

    if inputs is not None:
        prompt = prompt.format(**inputs)

    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx+1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."

    return prompt


def sample_machine_instructions(machine_instructions: List[str], n: int) -> List[str]:
    """Sample n machine instructions from a list of machine instructions."""

    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def extract_instructions(response: str):
    """Extracts the generated instructions from the response"""
    raw_instructions = re.split(r"\n\d+\d?\. ", response)
    generated_instructions = []

    for inst in raw_instructions:

        command = re.sub(r"\((.+)\)", " ", inst).strip().replace('"', "")
        generated_instructions.append(command)

    return generated_instructions


def format_preferences(preferences_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Format the preferences, remove empty fields"""

    for key, value in preferences_dict.items():
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError) as e:
                value = value.replace(";", ", ")

        if isinstance(value, list):
            preferences_dict[key] = ", ".join(value)

    return preferences_dict


def load_seed_instructions(seed_instruction_path: str) -> List[str]:
    """Load human-written instructions"""
    seed_instructions = []

    if seed_instruction_path is not None:

        seed_tasks = json.load(open(seed_instruction_path, "r"))

        seed_instructions = [
            t["instruction"] + " (" + t["metadata"]["devices"] + ")."
            for t in seed_tasks
        ]

    return seed_instructions


def load_machine_instructions(save_dir: str) -> Tuple[List[str], int]:
    """If the save folder contains saved instructions, load these instructions"""
    machine_instructions = []
    request_idx = 0

    if os.path.exists(save_dir):

        filenames = glob.glob(f"{save_dir}/*.json")

        for filename in filenames:
            instruction_info = json.load(open(filename))
            machine_instructions.append(instruction_info["instruction"])
            request_idx = request_idx + 1

    return machine_instructions, request_idx


def get_date():
    """Randomly generate a date"""
    start_date = datetime.date(2023, 8, 1)
    end_date = start_date + datetime.timedelta(days=10)

    random_date = start_date + (end_date - start_date) * random.random()

    return random_date


def select_instructions(
    scorer: rouge_scorer.RougeScorer,
    new_instructions: List[str],
    seed_instructions: List[str],
    machine_instructions: List[str],
):
    """Selects instructions based on how diverse they are"""

    selected_instructions = []

    for inst in new_instructions:
        with Pool(4) as p:
            rouge_scores = p.map(
                partial(scorer.score, inst),
                machine_instructions + seed_instructions,
            )
        rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]

        if len(rouge_scores) > 0:
            if max(rouge_scores) > 0.7:

                continue
        print(f"saving command {inst}")
        selected_instructions.append(inst)

    return selected_instructions


def generate_user_instructions(
    save_dir: str,
    user_name: str,
    seed_instruction_path: str,
    num_instructions: int,
    num_prompt_instructions: int,
    profile: Optional[Dict[str, Any]] = None,
    prompt_kwargs: Optional[Dict[str, Any]] = None,
    prompt_template: str = GENERIC_PROMPT,
):
    """Bootstrap instructions for a specific user"""

    os.makedirs(save_dir, exist_ok=True)

    if profile is not None:
        prompt_kwargs = {"preferences": profile}
        prompt_template = USER_PROFILE_PROMPT

        if not os.path.isfile(f"{os.path.dirname(save_dir)}/user_preferences.jsonl"):
            json.dump(
                profile,
                open(f"{os.path.dirname(save_dir)}/user_preferences.jsonl", "w"),
            )

    seed_instructions = load_seed_instructions(seed_instruction_path)
    machine_instructions, request_idx = load_machine_instructions(save_dir)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    dataset_writer = DatasetWriter(save_dir, False)

    progress_bar = tqdm.tqdm(total=num_instructions)

    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    while len(machine_instructions) < num_instructions:
        prompt_instructions = sample_machine_instructions(machine_instructions, n=3)

        if len(seed_instructions) > 0:
            prompt_instructions += random.sample(
                seed_instructions,
                num_prompt_instructions - len(prompt_instructions),
            )
        random.shuffle(prompt_instructions)

        prompt = encode_prompts(
            prompt_instructions,
            inputs=prompt_kwargs,
            prompt=prompt_template,
        )
        result = make_chatgpt_request(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=2,
            stop_sequences=["\n\n", "\n16", "16.", "16 ."],
            n=1,
        )

        new_instructions = extract_instructions(result["response"].content)
        selected_instructions = select_instructions(
            scorer, new_instructions, seed_instructions, machine_instructions
        )

        for inst in selected_instructions:
            machine_instructions.append(inst)

            dataset_writer.save_intermediate_result(
                {
                    "instruction": inst,
                    "request_idx": request_idx,
                    "date": str(get_date()),
                }
            )

            progress_bar.update(1)
            request_idx += 1
