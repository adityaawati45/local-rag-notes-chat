"""Step 5: search the vector database (retrieval only, no LLM yet).

    python search.py "What is the difference between ridge and lasso regression?"
    python search.py            (asks you for questions in a loop)
"""

import sys

import chromadb
import ollama

from build_index import COLLECTION_NAME, DB_PATH, EMBED_MODEL

QUERY_PREFIX = "search_query: "  # the partner of "search_document: " used when indexing
TOP_K = 4                        # how many chunks to retrieve


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION_NAME)


def search(collection, question, top_k=TOP_K):
    """Embed the question and return the top_k most similar chunks."""
    question_vector = ollama.embed(model=EMBED_MODEL, input=QUERY_PREFIX + question)["embeddings"][0]
    results = collection.query(query_embeddings=[question_vector], n_results=top_k)

    hits = []
    for text, meta, distance in zip(results["documents"][0], results["metadatas"][0],
                                    results["distances"][0]):
        hits.append({
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "similarity": 1 - distance,  # Chroma returns cosine *distance*; 1 - distance = similarity
        })
    return hits


def show(question, hits):
    print(f'\nQuestion: "{question}"\n')
    for rank, hit in enumerate(hits, start=1):
        print(f"#{rank}  similarity {hit['similarity']:.3f}  |  {hit['source']}, page {hit['page']}")
        print(f"    {hit['text'][:300]}...\n")


if __name__ == "__main__":
    collection = get_collection()
    print(f"Searching {collection.count()} chunks.")

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        show(question, search(collection, question))
    else:
        while True:
            question = input("\nAsk a question (or press Enter to quit): ").strip()
            if not question:
                break
            show(question, search(collection, question))
