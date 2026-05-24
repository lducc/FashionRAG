import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
collection_name = "fashion_products"
products_file = Path("data/products_sample.jsonl")
embeddings_file = Path("data/minilm_text_embeddings.npy")

products = []

with products_file.open("r") as file:
    for line in file:
        products.append(json.loads(line))

vectors = np.load(embeddings_file)
client = QdrantClient(url="http://localhost:6333")

if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE), #MiniLM vector is size 384
    )

points = []

for index, product in tqdm(list(enumerate(products)), desc="prepping points"):
    point = PointStruct(
        id=int(product["id"]),
        vector=vectors[index].tolist(),
        payload=product,
    )
    points.append(point)

batch_size = 100

for i in tqdm(range(0, len(points), batch_size), desc="uploading to qdrant"):
    batch = points[i:i + batch_size]
    client.upsert(collection_name=collection_name, points=batch)

print(f"Uploaded {len(points)} products to {collection_name}")
