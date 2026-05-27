from pathlib import Path


DATASET_NAME = "ashraq/fashion-product-images-small"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

PRODUCTS_FILE = Path("data/products_sample.jsonl")
CLIP_EMBEDDINGS_FILE = Path("data/clip_image_embeddings.npy")
BM25_FILE = Path("artifacts/bm25_vectorizer.pkl")

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "fashion_products"
CLIP_VECTOR_SIZE = 512
CLIP_VECTOR_NAME = "clip"
BATCH_SIZE = 64
