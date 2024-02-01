"""
This script will generate memories for multiple users based on their profiles
"""
import os
import dataclasses
from typing import Tuple

import glob
import tyro
from tyro.conf import UseAppendAction

from rouge_score import rouge_scorer

from sage.retrieval.data_generator.bootstrap_instructions import (
    format_preferences,
    generate_user_instructions,
)
from sage.retrieval.memory_bank import MemoryBank
from sage.utils.common import read_json, CONSOLE, check_env_vars


@dataclasses.dataclass
class GenerationConfig:

    # Path to the seed (human written) instructions
    seed_instruction_path: str = None
    # Where to save the dataset
    save_dir: str = f"{os.getenv('SMARTHOME_ROOT')}/data/memory_data"
    # The memory filename
    filename: str = "large_memory_bank"
    # The number of instructions to generate
    num_instructions_to_generate: int = 100
    # The number of example instructions to add to the context
    # The example instructions are taken from the seed and already generated
    # instructions
    num_prompt_instructions: int = 8
    # The path to the user profiles
    user_info_path: str = f"{os.getenv('SMARTHOME_ROOT')}/data/user_info"
    # Where to save generated instructions
    instruction_dir: str = "instructions"
    # User names
    user_names: UseAppendAction[Tuple[str, ...]] = ()
    # saved memories to inject
    saved_memory_path: str = (
        f"{os.getenv('SMARTHOME_ROOT')}/data/memory_data/manual_memories.json"
    )
    # The random seed
    seed: int = 42


def generate_instructions(config: GenerationConfig) -> None:
    """Generate instructions using an ChatGPT"""

    if not os.path.isdir(config.user_info_path):
        raise ValueError("Unvalid user info path")
    os.makedirs(config.save_dir, exist_ok=True)

    registered_users = [
        f.split("/")[-1] for f in glob.glob(f"{config.user_info_path}/*")
    ]
    user_names = list(set(registered_users).intersection(set(config.user_names)))

    for user_name in user_names:
        save_dir = os.path.join(
            config.user_info_path, f"{user_name}", config.instruction_dir
        )

        if os.path.isdir(save_dir) and os.listdir(save_dir):
            CONSOLE.log(f"[green]Instruction folder is not empty for user {user_name}")
            n = max(
                config.num_instructions_to_generate - len(os.listdir(save_dir)),
                0,
            )
            CONSOLE.log(f"[green]If you proceed, {n} instructions will be added")
            answer = input("Proceed ? Enter yes or no: ")

            if answer == "no":
                CONSOLE.log(f"[red]User {user_name} skipped")

                continue

        profile = read_json(
            f"{config.user_info_path}/{user_name}/user_preferences.jsonl"
        )

        # preprocess the profiles
        profile = format_preferences(profile)

        CONSOLE.log(
            f"[green]Generating memories for {user_name} using the profile {profile}"
        )

        generate_user_instructions(
            save_dir=save_dir,
            user_name=user_name,
            seed_instruction_path=config.seed_instruction_path,
            num_instructions=config.num_instructions_to_generate,
            num_prompt_instructions=config.num_prompt_instructions,
            profile=profile,
        )


def main(config: GenerationConfig) -> None:
    """
    Main function to generate the memory bank for multiple users.
    This is done as follows:
    1- If a memory bank already exists, add manual memories only.
    2- If the memory bank does not exist, generate new memories and add manual memories
    """

    check_env_vars()

    memory_bank_path = os.path.join(config.save_dir, f"{config.filename}.json")

    memory = MemoryBank()
    # check if memory bank exists

    if os.path.isfile(memory_bank_path):
        CONSOLE.log(
            f"[green]{memory_bank_path} already exists. Skipping instruction bootstrapping..."
        )
        memory.load(memory_bank_path)

    else:
        # Bootstrap instructions
        generate_instructions(config)
        # Generate the memory bank
        # This will load the json files for each user
        memory.load(config.user_info_path)
    print(memory)

    if config.saved_memory_path is not None:
        # manually inject saved memories
        saved_memories = read_json(config.saved_memory_path)

        for query in saved_memories:
            if not memory.contains(query["value"], query["user"]):
                memory.add_query(
                    user_name=query["user"], query=query["value"], date=query["date"]
                )

    print(len(memory))
    # Save the memory bank. This will automatically generate the user profiles
    memory.save(memory_bank_path)


if __name__ == "__main__":
    main(tyro.cli(GenerationConfig))
