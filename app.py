"""Step 7: a chat web app for your notes.

    streamlit run app.py

Opens in your browser at http://localhost:8501
"""

import ollama
import streamlit as st

from rag import CHAT_MODEL, SYSTEM_PROMPT, build_prompt
from search import get_collection, search

st.set_page_config(page_title="Chat with my notes", page_icon="📚")


@st.cache_resource  # open the database once, not on every click
def load_collection():
    return get_collection()


def stream_answer(messages):
    """Yield the LLM's answer piece by piece so it appears as it is written."""
    stream = ollama.chat(model=CHAT_MODEL, messages=messages, stream=True,
                         options={"num_ctx": 4096})
    for part in stream:
        yield part["message"]["content"]


def show_sources(sources):
    """A fold-out list of the passages the answer was based on."""
    with st.expander(f"Sources ({len(sources)})"):
        for number, hit in enumerate(sources, start=1):
            st.markdown(f"**[{number}] {hit['source']}**, page {hit['page']} "
                        f"· similarity {hit['similarity']:.2f}")
            st.caption(hit["text"][:400] + "...")


collection = load_collection()

# ---------- Sidebar: settings ----------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=8, value=4)
    min_similarity = st.slider(
        "Minimum similarity", min_value=0.0, max_value=1.0, value=0.65, step=0.01,
        help="Passages below this score are ignored. If none are left, the model is not asked "
             "at all, so it cannot make something up.",
    )
    st.caption(f"{collection.count()} chunks indexed · model: {CHAT_MODEL}")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Main area: the chat ----------
st.title("📚 Chat with my notes")
st.caption("Answers come only from your indexed course notes, with sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit reruns this whole script on every interaction, so redraw the history first
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

question = st.chat_input("Ask something about your notes...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your notes..."):
            hits = search(collection, question, top_k=top_k)                   # 1. Retrieval
        relevant = [hit for hit in hits if hit["similarity"] >= min_similarity]

        if not relevant:
            best = hits[0]["similarity"] if hits else 0.0
            answer = (f"I couldn't find anything relevant in your notes "
                      f"(best match {best:.2f}, below the {min_similarity:.2f} threshold).")
            st.markdown(answer)
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, relevant)},  # 2. Augmentation
            ]
            answer = st.write_stream(stream_answer(messages))                   # 3. Generation
            show_sources(relevant)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": relevant})
