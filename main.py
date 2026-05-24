import json
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


dataset_name = "ashraq/fashion-product-images-small"
output_file = Path("data/products_sample.jsonl")
embeddings_file = Path("data/minilm_text_embeddings.npy")
model_name = "sentence-transformers/all-MiniLM-L6-v2"

products = []

if output_file.exists():
    with output_file.open("r") as file:
        for line in tqdm(file, desc="reading products"):
            products.append(json.loads(line))
else:
    dataset = load_dataset(dataset_name, split="train")

    for row in tqdm(dataset, desc="cleaning products"):
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
        for product in tqdm(products, desc="saving products"):
            file.write(json.dumps(product) + "\n")

    print(f"Saved {len(products)} products to {output_file}")

demo_query = input("Enter a query: ")

print(f"\nSearch: {demo_query}")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
model.to(device)

if embeddings_file.exists():
    product_vectors = np.load(embeddings_file)
else:
    text_list = []

    for product in products:
        text_list.append(product["search_text"])

    vectors = []
    dataloader = DataLoader(text_list, batch_size=64)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Making text embeddings"):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1)
            text_features = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            vectors.append(text_features.cpu().numpy())

    product_vectors = np.vstack(vectors)
    np.save(embeddings_file, product_vectors)

with torch.no_grad():
    inputs = tokenizer([demo_query], padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    mask = attention_mask.unsqueeze(-1)
    query_vector = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
    query_vector = query_vector / query_vector.norm(dim=1, keepdim=True)
    query_vector = query_vector.cpu().numpy()[0]

scores = product_vectors @ query_vector
best_indexes = np.argsort(scores)[::-1]
seen_names = []

for index in best_indexes:
    product = products[index]

    if product["name"] in seen_names:
        continue

    seen_names.append(product["name"])
    product["score"] = round(float(scores[index]), 3)

    print(f"{product['name']} | {product['color']} | "
            f"{product['gender']} | score={product['score']}")

    if len(seen_names) == 7:
        break
