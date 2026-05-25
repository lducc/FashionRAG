import numpy as np

from datasets import load_dataset
from tqdm import tqdm

from fashionrag.embeddings import embed_texts
from fashionrag.products import build_product, save_products
from fashionrag.settings import DATASET_NAME, EMBEDDINGS_FILE, PRODUCTS_FILE


products = []
dataset = load_dataset(DATASET_NAME, split="train")

for row in tqdm(dataset, desc="Collecting data"):
    product = build_product(row)
    products.append(product)

save_products(products, PRODUCTS_FILE)
print(f"Saved {len(products)} products to {PRODUCTS_FILE}")

text_list = []

for product in products:
    text_list.append(product["search_text"])

product_vectors = embed_texts(text_list)
np.save(EMBEDDINGS_FILE, product_vectors)

print(f"Saved embeddings to {EMBEDDINGS_FILE}")
