# Chat with my notes: a local RAG app

Ask questions about your own documents (PDFs, Jupyter notebooks, text files) and get answers
**with sources**, from an LLM running entirely on your own laptop. No API keys, no cost,
and your documents never leave your machine.

Built from scratch to learn how **Retrieval-Augmented Generation (RAG)** works, step by step.

## Why RAG?

A small LLM on its own makes things up. Asked *"What is RAG in AI?"*, the model invented a
different meaning each time it was asked, once *"Representational Learning in Autonomous Systems"*,
once *"Reactive Attention Generator"*.

With RAG, the model answers **only from passages retrieved from your documents**:

| Question | Without RAG | With RAG |
|---|---|---|
| Difference between ridge and lasso regression? | Generic answer from memory | Correct answer citing the exact book pages [1][2] |
| What is RAG in AI? (not in the notes) | Confident hallucination | *"I couldn't find that in your notes."* |

## How it works

```
Indexing (once)   documents -> chunks -> embeddings -> Chroma vector DB
Answering         question -> embedding -> top-k similar chunks -> prompt + chunks -> LLM -> answer + sources
```

1. **Load and chunk** (`load_docs.py`): reads PDFs, notebooks, `.txt` and `.md` files from a folder and its
   subfolders, skips image-only pages and datasets, and splits the text into overlapping 200-word chunks.
2. **Embed and index** (`build_index.py`): turns every chunk into a 768-number vector with
   `nomic-embed-text` and stores it in a persistent Chroma collection (cosine similarity). It can be stopped and resumed.
3. **Retrieve** (`search.py`): embeds the question and finds the most similar chunks.
4. **Generate** (`rag.py`): builds a prompt with numbered passages and asks `llama3.2:3b` to answer
   only from them, citing sources.
5. **Chat app** (`app.py`): a Streamlit web interface with streaming answers, expandable sources and a
   similarity threshold that skips the LLM entirely when nothing relevant is found.

## Tech stack

Python · [Ollama](https://ollama.com) (local LLM and embeddings) · [Chroma](https://www.trychroma.com) (vector database) ·
Streamlit (web UI) · pypdf

## Setup

Requirements: Python 3.11+, [Ollama](https://ollama.com), about 3 GB of disk space for the models.

```bash
# 1. Get the code
git clone https://github.com/YOUR-USERNAME/my-rag.git
cd my-rag

# 2. Create a virtual environment and install the libraries
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 3. Download the models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Usage

Put your documents in a `docs/` folder, or point the app at any folder:

```bash
# Windows (PowerShell)
$env:RAG_DOCS_DIR = "D:\path\to\your\notes"
```

Then:

```bash
python load_docs.py          # preview how your documents are chunked
python build_index.py        # build the vector database (add --reset to rebuild from scratch)
python search.py             # semantic search only
python rag.py --compare "What is overfitting?"   # answer with vs. without RAG
streamlit run app.py         # the chat app, at http://localhost:8501
```

## Learning files

- `hello_llm.py`: a first call to a local LLM
- `embeddings_demo.py`: embeddings and cosine similarity written by hand

## Roadmap

- [ ] Follow-up questions (query rewriting with chat history)
- [ ] Hybrid search (keyword BM25 + vectors) and re-ranking
- [ ] Evaluation set to measure retrieval and answer quality
- [ ] OCR for image-only slides
- [ ] Optional cloud LLM for better answers
