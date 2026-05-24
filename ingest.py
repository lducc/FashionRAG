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

dataset = load_dataset(dataset_name, split="train")

for row in tqdm(dataset, desc="Collecting data"):
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

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
model.to(device)

text_list, vectors = [], []

for product in products:
    text_list.append(product["search_text"])

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

print(f"Saved embeddings to {embeddings_file}")
