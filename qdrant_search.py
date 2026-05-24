import numpy as np
import pandas as pd
import torch

from qdrant_client import QdrantClient
from transformers import AutoModel, AutoTokenizer


collection_name = "fashion_products"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
search_limit = 7

query = input("Enter a query: ")

print(f"\nSearch: {query}")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
model.to(device)

with torch.no_grad():
    inputs = tokenizer([query], padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    mask = attention_mask.unsqueeze(-1)
    query_vector = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
    query_vector = query_vector / query_vector.norm(dim=1, keepdim=True)
    query_vector = query_vector.cpu().numpy()[0]

client = QdrantClient(url="http://localhost:6333")

search_result = client.query_points_groups(
    collection_name=collection_name,
    query=query_vector.tolist(),
    group_by="name",
    limit=search_limit,
    group_size=1,
    with_payload=True,
)

for group in search_result.groups:
    point = group.hits[0]
    product = point.payload
    score = round(float(point.score), 3)

    print(f"{product['name']} | {product['color']} | "
            f"{product['gender']} | score={score}")
