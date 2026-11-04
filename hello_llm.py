import ollama

MODEL = "llama3.2:3b"

response = ollama.chat(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful teacher. Keep answers short."},
        {"role": "user", "content": "What is RAG in AI, in 3 sentences?"},
    ],
)

print(response["message"]["content"])