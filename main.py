import json
import numpy as np
import pandas as pd

from pathlib import Path
from datasets import load_dataset

dataset_name = "ashraq/fashion-product-images-small"
output_file = Path("data/products_sample.jsonl")

products = []

if output_file.exists():
    print(f"Loading saved products from {output_file}")

    with output_file.open("r") as file:
        for line in file:
            products.append(json.loads(line))
else:
    print("Downloading dataset from Hugging Face...")
    dataset = load_dataset(dataset_name, split="train")

    for row in dataset:
        product = {
            "id": str(row["id"]),
            "name": row["productDisplayName"],
            "category": row["masterCategory"],
            "subcategory": row["subCategory"],
            "article_type": row["articleType"],
            "color": row["baseColour"],
            "season": row["season"],
            "gender": row["gender"],
            "usage": row["usage"],
            "year": int(row["year"]),
        }

        product["search_text"] = (product["name"] + " "
                                + product["category"] + " "
                                + product["subcategory"] + " "
                                + product["article_type"]+ " "
                                + product["color"] + " "
                                + product["season"] + " "
                                + product["gender"] + " "
                                + product["usage"]).lower()
        products.append(product)

    output_file.parent.mkdir(exist_ok=True)

    with output_file.open("w") as file:
        for product in products:
            file.write(json.dumps(product) + "\n")

    print(f"Saved {len(products)} products to {output_file}")

demo_query = input("Enter a query: ")

print(f"\nSearch: {demo_query}")
words = demo_query.lower().split()
results = []

for product in products:
    score = 0

    for word in words:
        if word in product["search_text"]:
            score += 1

    if score > 0:
        product["score"] = score
        results.append(product)

results.sort(key = lambda item: item["score"], reverse=True)

for product in results[:7]:
    print(f"{product['name']} | {product['color']} | "
            f"{product['gender']} | score={product['score']}")
