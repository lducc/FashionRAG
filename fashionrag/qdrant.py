import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from fashionrag.embeddings import embed_texts
from fashionrag.products import load_products
from fashionrag.settings import (
    EMBEDDINGS_FILE,
    QDRANT_COLLECTION,
    QDRANT_URL,
    VECTOR_SIZE,
)


def get_client():
    return QdrantClient(url=QDRANT_URL)


def upload_products():
    products = load_products()
    vectors = np.load(EMBEDDINGS_FILE)
    client = get_client()

    if not client.collection_exists(collection_name=QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
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
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)

    client.close()
    return len(points)


def search_qdrant(query, limit=7):
    query_vector = embed_texts([query], batch_size=1, desc=None)[0]
    client = get_client()

    search_result = client.query_points_groups(
        collection_name=QDRANT_COLLECTION,
        query=query_vector.tolist(),
        group_by="name",
        limit=limit,
        group_size=1,
        with_payload=True,
    )

    results = []

    for group in search_result.groups:
        point = group.hits[0]
        product = point.payload
        product["score"] = round(float(point.score), 3)
        results.append(product)

    client.close()
    return results
