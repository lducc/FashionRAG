from pathlib import Path


DATASET_NAME = "ashraq/fashion-product-images-small"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

PRODUCTS_FILE = Path("data/products_sample.jsonl")
EMBEDDINGS_FILE = Path("data/minilm_text_embeddings.npy")
CLIP_EMBEDDINGS_FILE = Path("data/clip_image_embeddings.npy")
BM25_FILE = Path("artifacts/bm25_vectorizer.pkl")

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "fashion_products"
VECTOR_SIZE = 384
CLIP_VECTOR_SIZE = 512
TEXT_VECTOR_NAME = "text"
CLIP_VECTOR_NAME = "clip"
BATCH_SIZE = 64
