# FashionRAG

A simple fashion product search project using the `ashraq/fashion-product-images-small`
dataset and text embeddings.

## Setup

Install the project dependencies:

```bash
uv sync
```

## Build The Local Data

Run ingestion first. This downloads the fashion dataset, cleans product fields, and saves
the text embeddings.

```bash
uv run python ingest.py
```

This creates:

```text
data/products_sample.jsonl
data/minilm_text_embeddings.npy
```

## Search

After ingestion finishes, run the search script:

```bash
uv run python main.py
```

Then type a query like:

```text
brown shoes men
black watch women
blue casual shirt
```

## Notes

- The current search uses `sentence-transformers/all-MiniLM-L6-v2` through Transformers.
- `torchvision` is not needed for this version.
- If you want to check whether PyTorch sees your GPU:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```
