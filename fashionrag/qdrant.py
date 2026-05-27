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
    QDRANT_COLLECTION,
    QDRANT_URL,
)

UPLOAD_BATCH_SIZE = 100


def get_client():
    return QdrantClient(url=QDRANT_URL)


def vector_config():
    return {
        CLIP_VECTOR_NAME: VectorParams(size=CLIP_VECTOR_SIZE, distance=Distance.COSINE),
    }


def build_point(product, clip_vector):
    return PointStruct(
        id=int(product["id"]),
        vector={
            CLIP_VECTOR_NAME: clip_vector.tolist(),
        },
        payload=product,
    )


def upload_products():
    products = load_products()
    clip_vectors = np.load(CLIP_EMBEDDINGS_FILE)
    client = get_client()

    if len(products) != len(clip_vectors):
        raise ValueError("Products and embeddings have different lengths.")

    if client.collection_exists(collection_name=QDRANT_COLLECTION):
        client.delete_collection(collection_name=QDRANT_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=vector_config(),
    )

    points = []

    for index, product in tqdm(list(enumerate(products)), desc="prepping points"):
        points.append(build_point(product, clip_vectors[index]))

    for i in tqdm(range(0, len(points), UPLOAD_BATCH_SIZE), desc="uploading to qdrant"):
        batch = points[i:i + UPLOAD_BATCH_SIZE]
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
