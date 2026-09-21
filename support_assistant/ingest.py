import os
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = "support_assistant/docs"
CHROMA_PATH = "support_assistant/chroma_db"

def build_vector_store():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    try:
        client.delete_collection(name="zepto_policies")
    except:
        pass

    collection = client.create_collection(
        name="zepto_policies",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    docs, ids, metadatas = [], [], []
    for fname in sorted(os.listdir(DOCS_DIR)):
        if fname.endswith(".txt"):
            with open(os.path.join(DOCS_DIR, fname), "r", encoding="utf-8") as f:
                content = f.read().strip()
                docs.append(content)
                ids.append(fname.replace(".txt", ""))
                metadatas.append({"source": fname})

    collection.add(documents=docs, ids=ids, metadatas=metadatas)
    print(f"Indexed {len(docs)} documents into ChromaDB successfully.")

if __name__ == "__main__":
    build_vector_store()