import numpy as np

from datasets import load_dataset
from tqdm import tqdm

from fashionrag.clip import embed_images
from fashionrag.embeddings import embed_texts
from fashionrag.keyword import save_keyword_index
from fashionrag.products import build_product, save_products
from fashionrag.settings import (
    BM25_FILE,
    CLIP_EMBEDDINGS_FILE,
    DATASET_NAME,
    EMBEDDINGS_FILE,
    PRODUCTS_FILE,
)


def image_list(dataset):
    for row in dataset:
        yield row["image"]


def main():
    products = []
    dataset = load_dataset(DATASET_NAME, split="train")

    for row in tqdm(dataset, desc="Collecting data"):
        product = build_product(row)
        products.append(product)

    save_products(products, PRODUCTS_FILE)
    print(f"Saved {len(products)} products to {PRODUCTS_FILE}")

    save_keyword_index(products, BM25_FILE)
    print(f"Saved BM25 index to {BM25_FILE}")

    text_list = []

    for product in products:
        text_list.append(product["search_text"])

    product_vectors = embed_texts(text_list)
    np.save(EMBEDDINGS_FILE, product_vectors)

    print(f"Saved embeddings to {EMBEDDINGS_FILE}")

    clip_vectors = embed_images(image_list(dataset), total=len(dataset))
    np.save(CLIP_EMBEDDINGS_FILE, clip_vectors)

    print(f"Saved CLIP image embeddings to {CLIP_EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
