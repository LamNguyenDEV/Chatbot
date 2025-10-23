import os
import nltk
from sentence_transformers import SentenceTransformer
import chromadb

# Note
# PREPARE_DATA IS PART 1, RETRIEVE_DATA IS PART 2 OF THIS PROGRAM
# Prepare_data.py purpose for Set Up Retrieval (Vector Database).
print(" Starting ingestion process...")

# === Download NLTK punkt tokenizer for sentence splitting (only first run) ===
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# === Step 1: Read Markdown files ===
def read_markdown_files(folder):
    texts = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".md"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as file:
                    text = file.read()
                # remove metadata (front matter)
                if text.startswith("---"):
                    text = text.split("---", 2)[-1].strip()
                texts.append((path, text))
    return texts

texts = read_markdown_files("data/processed")
print(f"✅ Loaded {len(texts)} markdown files.")

# === Step 2: Improved chunking by sentences with overlap ===
def chunk_by_sentences(text, max_words=400, overlap=100):
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_count = [], [], 0

    for sent in sentences:
        words = sent.split()
        current_chunk.append(sent)
        current_count += len(words)

        if current_count >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_count = sum(len(s.split()) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

all_chunks = []
for path, text in texts:
    for i, chunk in enumerate(chunk_by_sentences(text, max_words=400, overlap=100)):
        all_chunks.append({"source": path, "chunk_id": i, "text": chunk})

print(f"✅ Created {len(all_chunks)} chunks total.")

# === Step 3: Load embedding model ===
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# === Step 4: Embed chunks ===
texts_for_embedding = [c["text"] for c in all_chunks]
embeddings = model.encode(texts_for_embedding, show_progress_bar=True)

for c, emb in zip(all_chunks, embeddings):
    c["embedding"] = emb

print("✅ Finished embeddings successfully.")

# === Step 5: Store in persistent ChromaDB ===
persist_directory = "vector_store_chroma"
os.makedirs(persist_directory, exist_ok=True)

client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_or_create_collection(name="website_docs")

ids = [f"{c['source']}_{c['chunk_id']}" for c in all_chunks]
metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in all_chunks]
documents = [c["text"] for c in all_chunks]
embeddings_list = [c["embedding"].tolist() for c in all_chunks]

batch_size = 100
print("Adding chunks to ChromaDB...")
for i in range(0, len(all_chunks), batch_size):
    collection.add(
        ids=ids[i:i+batch_size],
        documents=documents[i:i+batch_size],
        embeddings=embeddings_list[i:i+batch_size],
        metadatas=metadatas[i:i+batch_size]
    )

print(f"✅ All {len(all_chunks)} chunks stored in ChromaDB at '{persist_directory}'")
print("🎉 Ingestion complete. Run `query_data.py` to test retrieval.")
