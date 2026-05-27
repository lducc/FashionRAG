import numpy as np

from fashionrag.qdrant import build_point, vector_config
from fashionrag.settings import CLIP_VECTOR_NAME


def test_vector_config_has_clip_vector():
    config = vector_config()

    assert config[CLIP_VECTOR_NAME].size == 512


def test_build_point_uses_clip_vector():
    product = {"id": "123", "name": "Blue Shirt"}
    clip_vector = np.ones(512)

    point = build_point(product, clip_vector)

    assert point.id == 123
    assert point.payload == product
    assert len(point.vector[CLIP_VECTOR_NAME]) == 512
