import math
import ollama

EMBED_MODEL = "nomic-embed-text"

# A tiny "knowledge base" of sentences
sentences = [
    "How do I reset my password?",
    "Steps to recover a forgotten login.",
    "What is the weather like today?",
    "Is it going to rain tomorrow?",
    "The best pizza in Berlin is near Alexanderplatz.",
]


def cosine_similarity(a, b):
    """How similar two vectors are: close to 1 = same meaning, lower = less related."""
    dot = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    return dot / (length_a * length_b)


# --- Part 1: turn every sentence into a vector ---
vectors = ollama.embed(model=EMBED_MODEL, input=sentences)["embeddings"]

print(f"Each sentence became a list of {len(vectors[0])} numbers.")
print("The first 5 numbers of sentence 1:", [round(x, 3) for x in vectors[0][:5]])
print()

# --- Part 2: a mini semantic search ---
question = "I can't log into my account"
question_vector = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]

scored = []
for sentence, vector in zip(sentences, vectors):
    score = cosine_similarity(question_vector, vector)
    scored.append((score, sentence))

scored.sort(reverse=True)  # highest similarity first

print(f'Question: "{question}"')
print("Most similar sentences:")
for score, sentence in scored:
    print(f"  {score:.3f}  {sentence}")