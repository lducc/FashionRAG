from pathlib import Path
DATASET_NAME = "ashraq/fashion-product-images-small"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PRODUCTS_FILE = Path("data/products_sample.jsonl")
EMBEDDINGS_FILE = Path("data/minilm_text_embeddings.npy")
BM25_FILE = Path("artifacts/bm25_vectorizer.pkl")

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "fashion_products"
VECTOR_SIZE = 384
BATCH_SIZE = 64
