import json
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from transformers import AutoModel, AutoTokenizer


output_file = Path("data/products_sample.jsonl")
embeddings_file = Path("data/minilm_text_embeddings.npy")
model_name = "sentence-transformers/all-MiniLM-L6-v2"

products = []

with output_file.open("r") as file:
    for line in file:
        products.append(json.loads(line))

product_vectors = np.load(embeddings_file)

demo_query = input("Enter a query: ")

print(f"\nSearch: {demo_query}")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
model.to(device)

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
