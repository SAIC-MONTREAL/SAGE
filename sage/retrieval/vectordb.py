"""
Everything related to vectordbs
"""
import os
from typing import List, Dict
from pathlib import Path
import shutil
from langchain.vectorstores import Chroma, FAISS
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings


from sage.utils.common import CONSOLE


def build_chroma_db(
    vector_dir: str, documents: List[Document], embeddings: List[Embeddings], load=True
) -> Chroma:
    """Creates or loads a chroma database"""

    if os.path.isdir(vector_dir) and load is True:
        CONSOLE.log(f"Loading vector db from {vector_dir}....")

        return Chroma(
            persist_directory=vector_dir,
            embedding_function=embeddings,
        )

    if not load:
        if os.path.isdir(vector_dir):
            shutil.rmtree(vector_dir)
            CONSOLE.log("Existing db wiped!")
        CONSOLE.log(f"Creating vector db in {vector_dir}...")

    return Chroma.from_documents(documents, embeddings, persist_directory=vector_dir)


def build_faiss_db(
    vector_dir,
    documents: List[Document],
    embeddings: List[Embeddings],
    load: bool = True,
):
    """Creates or loads a FAISS index"""

    if os.path.isdir(vector_dir) and load:
        CONSOLE.log(f"Loading vector db from {vector_dir}....")
        index = FAISS.load_local("smartie-index", embeddings)

        return index

    if not load:
        files = Path(vector_dir).glob("sage-index.*")

        if files:
            for filename in files:
                filename.unlink()

    CONSOLE.log("Creating vector db ...")
    index = FAISS.from_documents(documents=documents, embedding=embeddings)
    index.save_local(folder_path=vector_dir, index_name="sage-index")

    return index


VECTORDBS = {"chroma": build_chroma_db, "faiss": build_faiss_db}


def create_multiuser_vector_indexes(
    vectordb: str,
    documents: Dict[str, List[Document]],
    embedding_model,
    load: bool = True,
):
    """Creates a vector index that offers similarity search"""

    user_indexes = {}

    for user_name, memories in documents.items():
        user_index_dir = os.path.join(
            f"{os.getenv('SMARTHOME_ROOT')}", "user_info", user_name, vectordb
        )

        # Create the index
        user_indexes[user_name] = VECTORDBS[vectordb](
            user_index_dir, memories, embedding_model, load=load
        )

    return user_indexes
