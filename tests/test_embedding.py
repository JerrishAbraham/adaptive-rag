from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

texts = [
    "Retrieval-Augmented Generation combines retrieval with language generation.",
    "FAISS is used for efficient similarity search."
]

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)
print("Embedding model test successful!")