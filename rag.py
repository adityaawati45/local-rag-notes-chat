"""Step 6: the full RAG pipeline: retrieve relevant chunks, then let the LLM answer from them.

    python rag.py                          (asks you for questions in a loop)
    python rag.py "What is overfitting?"
    python rag.py --compare "What is overfitting?"   (also shows the answer WITHOUT RAG)
"""

import sys

import ollama

from search import get_collection, search

CHAT_MODEL = "llama3.2:3b"
TOP_K = 4  # how many chunks to give the LLM

SYSTEM_PROMPT = """You are a study assistant that answers questions about the user's course notes.

Rules:
- Answer ONLY with information from the numbered context passages you are given.
- If the passages do not contain the answer, reply exactly: "I couldn't find that in your notes."
- Cite the passages you used with their numbers, like [1] or [2][3].
- Be clear and concise. Use short paragraphs or bullet points."""


def build_prompt(question, hits):
    """The 'augmented' prompt: numbered context passages followed by the question."""
    passages = []
    for number, hit in enumerate(hits, start=1):
        passages.append(f"[{number}] (from {hit['source']}, page {hit['page']})\n{hit['text']}")
    context = "\n\n".join(passages)
    return f"Context passages:\n\n{context}\n\nQuestion: {question}"


def generate(messages):
    """Send messages to the LLM and print the answer word by word as it is generated."""
    answer = ""
    stream = ollama.chat(model=CHAT_MODEL, messages=messages, stream=True,
                         options={"num_ctx": 4096})  # room for the question plus 4 passages
    for part in stream:
        piece = part["message"]["content"]
        print(piece, end="", flush=True)
        answer += piece
    print()
    return answer


def ask(collection, question):
    """Retrieve, augment, generate. Returns the answer and the chunks it was based on."""
    hits = search(collection, question, top_k=TOP_K)          # 1. Retrieval
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, hits)},  # 2. Augmentation
    ]
    print("\nAnswer (with your notes):\n")
    answer = generate(messages)                                # 3. Generation

    print("\nSources:")
    for number, hit in enumerate(hits, start=1):
        print(f"  [{number}] {hit['source']}, page {hit['page']}  (similarity {hit['similarity']:.2f})")
    return answer, hits


def ask_without_rag(question):
    """The same question with no notes: the model answers from memory only."""
    print("\nAnswer WITHOUT your notes (model's memory only):\n")
    return generate([{"role": "user", "content": question}])


if __name__ == "__main__":
    compare = "--compare" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--compare"]
    collection = get_collection()
    print(f"Ready. Searching {collection.count()} chunks of your notes with {CHAT_MODEL}.")

    questions = [" ".join(args)] if args else None
    while True:
        question = questions.pop() if questions else input("\nAsk a question (or press Enter to quit): ").strip()
        if not question:
            break
        if compare:
            ask_without_rag(question)
            print("\n" + "=" * 60)
        ask(collection, question)
        if args:  # a question was given on the command line: answer it once and stop
            break
