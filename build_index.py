"""Step 4: embed every chunk and store it in a Chroma vector database.

    python build_index.py              (indexes DOCS_DIR from load_docs.py)
    python build_index.py "D:\\some\\other\\folder"
    python build_index.py --reset      (delete the database first and rebuild)

Safe to stop with Ctrl+C and run again: chunks already stored are skipped.
"""

import sys
import time
from pathlib import Path

import chromadb
import ollama

from load_docs import DOCS_DIR, load_and_chunk

EMBED_MODEL = "nomic-embed-text"
DB_PATH = "chroma_db"        # folder inside the project where Chroma saves its data
COLLECTION_NAME = "notes"
BATCH_SIZE = 32              # chunks embedded per call to Ollama

# nomic-embed-text works best when told whether a text is a document or a search query
DOCUMENT_PREFIX = "search_document: "


def get_collection(reset=False):
    """Open (or create) the Chroma collection that holds our chunks."""
    client = chromadb.PersistentClient(path=DB_PATH)
    if reset and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted the old collection.")
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},  # compare vectors by cosine similarity
    )


def embed_documents(texts):
    """Turn a list of chunk texts into vectors with the embedding model."""
    response = ollama.embed(model=EMBED_MODEL, input=[DOCUMENT_PREFIX + t for t in texts])
    return response["embeddings"]


def build_index(docs_dir=DOCS_DIR, reset=False):
    docs_dir = Path(docs_dir)
    print(f"Loading documents from: {docs_dir}\n")
    chunks = load_and_chunk(docs_dir)

    # Prefix ids with the folder name so several modules can share one database
    for chunk in chunks:
        chunk["id"] = f"{docs_dir.name}/{chunk['id']}"

    collection = get_collection(reset)

    # Skip chunks that are already stored (lets you stop and resume)
    already_stored = set(collection.get(ids=[c["id"] for c in chunks], include=[])["ids"])
    todo = [c for c in chunks if c["id"] not in already_stored]
    print(f"\n{len(chunks)} chunks found, {len(already_stored)} already in the database, "
          f"{len(todo)} to embed.\n")

    start = time.time()
    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i:i + BATCH_SIZE]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embed_documents([c["text"] for c in batch]),
            metadatas=[{"source": c["source"], "page": c["page"], "folder": docs_dir.name}
                       for c in batch],
        )

        done = i + len(batch)
        elapsed = time.time() - start
        remaining = elapsed / done * (len(todo) - done)
        print(f"  {done}/{len(todo)} chunks embedded  "
              f"({elapsed:.0f}s so far, about {remaining:.0f}s left)")

    print(f"\nDone. The database now holds {collection.count()} chunks "
          f"(saved in the '{DB_PATH}' folder).")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--reset"]
    build_index(docs_dir=args[0] if args else DOCS_DIR, reset="--reset" in sys.argv)
