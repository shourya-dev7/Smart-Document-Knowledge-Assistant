from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Quantum computing uses quantum bits called qubits."

embedding = model.encode(text)

print("Embedding created successfully!")
print("Number of values:", len(embedding))
print("First 5 values:", embedding[:5])