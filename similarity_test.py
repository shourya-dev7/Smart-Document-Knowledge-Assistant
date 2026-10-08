from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence1 = "A qubit can exist in a superposition of states."
sentence2 = "Quantum bits can be in multiple states at the same time."
sentence3 = "The capital of France is Paris."

embeddings = model.encode([sentence1, sentence2, sentence3])

similarity_1_2 = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)[0][0]

similarity_1_3 = cosine_similarity(
    [embeddings[0]],
    [embeddings[2]]
)[0][0]

print("Similarity between sentence 1 and 2:", similarity_1_2)
print("Similarity between sentence 1 and 3:", similarity_1_3)