"""
This is where the memory bank is constructed and updated
The memory bank consists of the following components:
    - Interaction history
    - User preferences
"""
import os
import json
import glob
from typing import List
from collections import defaultdict
from langchain.schema.document import Document
from sage.retrieval.profiler import UserProfiler
from sage.retrieval.vectordb import create_multiuser_vector_indexes
from sage.utils.common import load_embedding_model


class MemoryBank:
    """This class handles the memory bank"""

    def __init__(self):
        self.history = defaultdict(dict)
        self.user_profiler = UserProfiler()
        self.indexes = defaultdict(list)
        self.snapshot_id = 0

    def _load_user_queries(self, user_name: str, directory: str):
        """
        Loads user queries from a directory.
        This assumes that the queries are saved in JSON files
        """

        if os.path.isdir(f"{directory}/instructions") and os.listdir(
            f"{directory}/instructions"
        ):

            filenames = glob.glob(f"{directory}/instructions/*.json")

            for filename in filenames:
                instruction_info = json.load(open(filename))
                self.add_query(
                    user_name, instruction_info["instruction"], instruction_info["date"]
                )

        else:
            print(f"The instructions folder is empty. Skipping user {user_name}")

    def load(self, memory_path: str):
        """Load memory bank from directory"""

        print(f"Loading memory from {memory_path}")

        if os.path.isfile(memory_path) and memory_path.endswith(".json"):
            # Load saved json file
            self.history = json.load(open(memory_path))

        elif os.path.isdir(memory_path):
            sources = glob.glob(f"{memory_path}/*")

            for source in sources:
                if os.path.isdir(source):
                    # This assumes that the memory is a list of directories
                    # where in each directory contains the interactions of
                    # one user
                    user_name = os.path.basename(source)
                else:
                    # This assumes that the interactions are not organized by user name
                    # give a fake username
                    user_name = "all"

                self._load_user_queries(user_name, source)
        else:
            raise ValueError(f"Invalid memory path {memory_path}. Please check ")

    def add_query(self, user_name: str, query: str, date: str):
        """Add a query to the history"""

        if self.history[user_name].get("history") is None:
            self.history[user_name] = {"history": defaultdict(list)}

        if self.history[user_name]["history"].get(date) is None:
            self.history[user_name]["history"][date] = []

        self.history[user_name]["history"][date].append(query)

    def _build_user_profiles(self):
        """Build the user profiles based on the saved interactions"""

        for user_name in self.history.keys():
            user_queries = self.history[user_name]["history"]

            for date, queries in user_queries.items():
                self.user_profiler.update_daily_user_preferences(
                    user_name, queries, date
                )

            self.user_profiler.create_global_user_profile(user_name)

            self.history[user_name]["profile"] = self.user_profiler.global_profiles[
                user_name
            ]
        self.user_profiler.print_global_profiles()

    def read_from_json(self, save_path: str) -> None:
        """Reads memory from json file"""
        self.history = json.load(open(save_path, "r"))

        for user_name, data in self.history.items():
            self.user_profiler.global_profiles[user_name] = data["profile"]

    def save(self, save_path: str):
        """Saves the memory into a json file"""

        self._build_user_profiles()

        json.dump(self.history, open(save_path, "w"))

    def save_snapshot(self, save_path: str):
        """Save a snapshot of the memory"""

        filename = save_path

        if not save_path.endswith("json"):
            filename = os.path.join(save_path, f"snapshot_{self.snapshot_id}.json")

        json.dump(
            self.history,
            open(filename, "w"),
        )
        self.snapshot_id += 1

    def prepare_for_vector_db(self):
        """
        Prepare documents from the users' history for embedding and vector storage
        """
        documents = defaultdict(list)

        for user_name, user_memory in self.history.items():
            for date, interactions in user_memory["history"].items():
                for instruction in interactions:
                    memory_text = f"Instruction on {date}: {instruction.strip()}"
                    metadata = {"source": date, "user": user_name}
                    documents[user_name].append(
                        Document(page_content=memory_text, metadata=metadata)
                    )

        return documents

    def create_indexes(
        self, vectorstore: str, embedding_model: str, load: bool = True
    ) -> None:
        """Create seperate indexes for each user"""
        documents = self.prepare_for_vector_db()
        emb_function = load_embedding_model(model_name=embedding_model)
        self.indexes = create_multiuser_vector_indexes(
            vectorstore, documents, emb_function, load=load
        )

    def search(self, user_name: str, query: str, top_k=5) -> List[str]:
        """Get the most relevant memories"""
        sources = self.indexes[user_name].similarity_search(query, k=top_k)

        memories = [s.page_content for s in sources]

        return memories

    def contains(self, memory: str, user_name: str) -> bool:
        """Check if a specific memory exists"""

        if user_name not in self.history:
            return False

        user_memories = self.history[user_name]["history"]

        for _, value in user_memories.items():
            if value == memory:
                return True

        return False

    def __len__(self):
        total = 0

        for user, data in self.history.items():
            user_total = 0

            for key, value in data["history"].items():
                user_total += len(value)

            print(f"User {user} has {user_total} saved memories")
            total += user_total

        return total
