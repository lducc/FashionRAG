import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from fashionrag.clip import embed_clip_texts
from fashionrag.products import load_products
from fashionrag.settings import (
    CLIP_EMBEDDINGS_FILE,
    CLIP_VECTOR_NAME,
    CLIP_VECTOR_SIZE,
    EMBEDDINGS_FILE,
    QDRANT_COLLECTION,
    QDRANT_URL,
    TEXT_VECTOR_NAME,
    VECTOR_SIZE,
)


def get_client():
    return QdrantClient(url=QDRANT_URL)


def upload_products():
    products = load_products()
    text_vectors = np.load(EMBEDDINGS_FILE)
    clip_vectors = np.load(CLIP_EMBEDDINGS_FILE)
    client = get_client()

    if len(products) != len(text_vectors) or len(products) != len(clip_vectors):
        raise ValueError("Products and embeddings have different lengths.")

    if client.collection_exists(collection_name=QDRANT_COLLECTION):
        client.delete_collection(collection_name=QDRANT_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            CLIP_VECTOR_NAME: VectorParams(size=CLIP_VECTOR_SIZE, distance=Distance.COSINE),
            TEXT_VECTOR_NAME: VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        },
    )

    points = []

    for index, product in tqdm(list(enumerate(products)), desc="prepping points"):
        point = PointStruct(
            id=int(product["id"]),
            vector={
                CLIP_VECTOR_NAME: clip_vectors[index].tolist(),
                TEXT_VECTOR_NAME: text_vectors[index].tolist(),
            },
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
    query_vector = embed_clip_texts([query], batch_size=1, desc=None)[0]
    client = get_client()

    search_result = client.query_points_groups(
        collection_name=QDRANT_COLLECTION,
        query=query_vector.tolist(),
        using=CLIP_VECTOR_NAME,
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
