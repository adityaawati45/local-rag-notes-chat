"""Step 3: load documents from a folder (and its subfolders) and split them into chunks.

Run it on its own to preview the chunks:
    python load_docs.py                      (uses DOCS_DIR below)
    python load_docs.py "D:\\some\\other\\folder"
"""

import json
import logging
import sys
from pathlib import Path

from pypdf import PdfReader

# Folder to index. We start with one module; point it at the whole folder later.
DOCS_DIR = Path(r"D:\NOTES\surya_pg_dbda-main\surya_pg_dbda-main\module_7_PracticalMachineLearning")

CHUNK_SIZE = 200          # words per chunk
OVERLAP = 40              # words shared between neighbouring chunks
MIN_WORDS_PER_PAGE = 30   # skip pages with almost no text (slide images, title pages)
MAX_TXT_SIZE = 1_000_000  # skip .txt files over 1 MB: in this folder those are datasets, not notes

SUPPORTED = {".pdf", ".ipynb", ".txt", ".md"}
SKIP_FOLDER_WORDS = ["dataset", ".metadata", ".ipynb_checkpoints", "__pycache__"]

logging.getLogger("pypdf").setLevel(logging.ERROR)  # hide noisy warnings from messy PDFs


def clean(text):
    """Collapse line breaks and repeated spaces into single spaces."""
    return " ".join(text.split())


def load_pdf(path):
    """Read a PDF and return one {page, text} item per page that has real text."""
    pages = []
    for page_number, page in enumerate(PdfReader(path).pages, start=1):
        text = clean(page.extract_text() or "")
        if len(text.split()) >= MIN_WORDS_PER_PAGE:
            pages.append({"page": page_number, "text": text})
    return pages


def load_notebook(path):
    """Read a Jupyter notebook: explanations (markdown cells) plus the code cells."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        if cell.get("cell_type") == "markdown":
            parts.append(source)
        elif cell.get("cell_type") == "code":
            parts.append("Code: " + source)
    text = clean("\n".join(parts))
    return [{"page": 1, "text": text}] if text else []


def load_txt(path):
    """Read a .txt or .md file as a single 'page'."""
    text = clean(path.read_text(encoding="utf-8", errors="ignore"))
    return [{"page": 1, "text": text}] if len(text.split()) >= MIN_WORDS_PER_PAGE else []


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping chunks of about chunk_size words."""
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def find_files(docs_dir):
    """Every supported file in docs_dir and its subfolders, minus the ones we skip."""
    files, skipped = [], []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        folders = [part.lower() for part in path.relative_to(docs_dir).parts[:-1]]
        if any(word in folder for folder in folders for word in SKIP_FOLDER_WORDS):
            skipped.append((path, "dataset or system folder"))
        elif path.suffix.lower() == ".pdf" and path.with_suffix(".ipynb").exists():
            skipped.append((path, "PDF copy of a notebook we already read"))
        elif path.suffix.lower() == ".txt" and path.stat().st_size > MAX_TXT_SIZE:
            skipped.append((path, "large .txt, probably data"))
        else:
            files.append(path)
    return files, skipped


def load_and_chunk(docs_dir=DOCS_DIR):
    """Load every document in docs_dir and return a list of chunks with metadata."""
    docs_dir = Path(docs_dir)
    files, skipped = find_files(docs_dir)
    all_chunks = []
    no_text = []

    for path in files:
        relative = path.relative_to(docs_dir).as_posix()
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                pages = load_pdf(path)
            elif suffix == ".ipynb":
                pages = load_notebook(path)
            else:
                pages = load_txt(path)
        except Exception as error:  # a broken file shouldn't stop the whole run
            print(f"  ERROR reading {relative}: {error}")
            continue

        if not pages:
            no_text.append(relative)
            continue

        count_before = len(all_chunks)
        for page in pages:
            for n, chunk in enumerate(chunk_text(page["text"])):
                all_chunks.append({
                    "id": f"{relative}-p{page['page']}-c{n}",
                    "text": chunk,
                    "source": relative,
                    "page": page["page"],
                })
        print(f"  {len(all_chunks) - count_before:5} chunks  {relative}")

    print(f"\nRead {len(files) - len(no_text)} files, skipped {len(skipped)} on purpose.")
    if no_text:
        print(f"{len(no_text)} files had no readable text (probably slides saved as images):")
        for relative in no_text:
            print(f"    - {relative}")
    return all_chunks


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DOCS_DIR
    print(f"Loading documents from: {folder}\n")
    chunks = load_and_chunk(folder)

    total_words = sum(len(c["text"].split()) for c in chunks)
    print(f"\nTotal chunks: {len(chunks)}  (about {total_words:,} words)")

    for chunk in chunks[:2]:
        print("\n" + "-" * 60)
        print(f"ID: {chunk['id']}")
        print(f"Source: {chunk['source']}, page {chunk['page']}")
        print(chunk["text"][:400] + "...")
