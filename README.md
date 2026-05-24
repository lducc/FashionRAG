# FashionRAG

A small fashion product search project using local embeddings and Qdrant.

The dataset is `ashraq/fashion-product-images-small` from Hugging Face. Product text is embedded
with `sentence-transformers/all-MiniLM-L6-v2`, then uploaded to a local Qdrant vector database.

## Setup

Install dependencies:

```bash
uv sync
```

## 1. Build The Data Files

Run ingestion:

```bash
uv run python ingest.py
```

This creates:

```text
data/products_sample.jsonl
data/minilm_text_embeddings.npy
```

## 2. Start Qdrant Locally

Run Qdrant with Docker:

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

The Qdrant dashboard should be available at:

```text
http://localhost:6333/dashboard
```

## 3. Upload To Qdrant

In another terminal, upload the local product data and embeddings:

```bash
uv run python qdrant_upload.py
```

This creates or reuses a collection named:

```text
fashion_products
```

## 4. Search Qdrant

Run:

```bash
uv run python qdrant_search.py
```

Try queries like:

```text
brown shoes men
black watch women
blue casual shirt
```

## Local Search

There is also a local NumPy search script:

```bash
uv run python main.py
```

This searches `data/minilm_text_embeddings.npy` directly instead of Qdrant.

## Notes

- Qdrant runs locally at `http://localhost:6333`.
- The vector size is `384` because MiniLM returns 384-dimensional embeddings.
- `qdrant_storage/` is local database storage and should not be committed.
- `torchvision` is not needed for this version.
