# FashionRAG

A small fashion product search project using local embeddings and Qdrant.

The dataset is `ashraq/fashion-product-images-small` from Hugging Face. Product text is embedded
with `sentence-transformers/all-MiniLM-L6-v2`, then uploaded to a local Qdrant vector database.

## Project Structure

```text
fashionrag/   # shared project code
scripts/      # runnable commands
data/         # generated products and embeddings
```

## Setup

Install dependencies:

```bash
uv sync
```

Optional Hugging Face token setup:

```bash
cp .env.example .env
```

Then add your token in `.env`:

```text
HF_TOKEN=hf_your_token_here
```

Use `--env-file .env` when running commands that download models or datasets.

## 1. Build The Data Files

Run ingestion. This downloads the dataset, cleans product fields, and creates embeddings:

```bash
uv run --env-file .env python -m scripts.ingest
```

This creates:

```text
data/products_sample.jsonl
data/minilm_text_embeddings.npy
artifacts/bm25_vectorizer.pkl
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

In another terminal, upload products and embeddings to Qdrant:

```bash
uv run --env-file .env python -m scripts.qdrant_upload
```

This creates or reuses a collection named:

```text
fashion_products
```

## 4. Search Qdrant

Run semantic search through Qdrant:

```bash
uv run --env-file .env python -m scripts.qdrant_search
```

Run hybrid keyword + semantic search:

```bash
uv run --env-file .env python -m scripts.hybrid_search
```

Run the retrieval eval:

```bash
uv run --env-file .env python -m eval.ablation
```

Try queries like:

```text
brown shoes men
black watch women
blue casual shirt
```

## App

Run the simple Gradio app:

```bash
uv run --env-file .env python -m scripts.app
```

## Local Search

There is also a local NumPy search script. This does not use Qdrant:

```bash
uv run --env-file .env python -m scripts.local_search
```

And a keyword-only BM25 script:

```bash
uv run --env-file .env python -m scripts.keyword_search
```

## Notes

- Qdrant runs locally at `http://localhost:6333`.
- The vector size is `384` because MiniLM returns 384-dimensional embeddings.
- Hybrid search combines Qdrant semantic search with BM25 keyword search.
- BM25 uses NLTK tokenization and stemming.
- The BM25 index is built during ingestion and loaded during search.
- The eval compares semantic-only retrieval against hybrid retrieval with Recall@5 and MRR.
- `qdrant_storage/` is local database storage and should not be committed.
- `torchvision` is not needed for this version.
