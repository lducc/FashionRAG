import numpy as np

from fashionrag.qdrant import build_point, vector_config
from fashionrag.settings import CLIP_VECTOR_NAME, TEXT_VECTOR_NAME


def test_vector_config_has_named_vectors():
    config = vector_config()

    assert config[CLIP_VECTOR_NAME].size == 512
    assert config[TEXT_VECTOR_NAME].size == 384


def test_build_point_uses_named_vectors():
    product = {"id": "123", "name": "Blue Shirt"}
    text_vector = np.zeros(384)
    clip_vector = np.ones(512)

    point = build_point(product, text_vector, clip_vector)

    assert point.id == 123
    assert point.payload == product
    assert len(point.vector[CLIP_VECTOR_NAME]) == 512
    assert len(point.vector[TEXT_VECTOR_NAME]) == 384
