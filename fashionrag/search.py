import numpy as np

from fashionrag.embeddings import embed_texts
from fashionrag.products import load_products
from fashionrag.settings import EMBEDDINGS_FILE
def local_search(query, limit=7):
    products = load_products()
    product_vectors = np.load(EMBEDDINGS_FILE)
    query_vector = embed_texts([query], batch_size=1, desc=None)[0]

    scores = product_vectors @ query_vector
    best_indexes = np.argsort(scores)[::-1]
    seen_names = []
    results = []

    for index in best_indexes:
        product = products[index]

        if product["name"] in seen_names:
            continue

        seen_names.append(product["name"])
        product["score"] = round(float(scores[index]), 3)
        results.append(product)

        if len(results) == limit:
            break

    return results
