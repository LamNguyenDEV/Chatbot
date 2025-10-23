from sentence_transformers import SentenceTransformer
import chromadb

print("🔍 Starting query session...")

# === Load embedding model ===
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# === Load persistent ChromaDB ===
persist_directory = "vector_store_chroma"
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("website_docs")

# === Encode a query ===
query = input("Enter your question: ")
query_emb = model.encode([query]).tolist()

# === Retrieve top 3 results ===
results = collection.query(
    query_embeddings=query_emb,
    n_results=3
)

print("\n=== Top 3 Results ===")
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), start=1):
    #print(f"\n{i}. Source: {meta['source']}")
    print(f"Snippet:\n{doc[:400]}...")
    print("---")
