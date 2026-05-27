import torch

from fashionrag.clip import get_vector, normalize


class FakeOutput:
    def __init__(self, pooler_output=None, text_embeds=None, image_embeds=None):
        self.pooler_output = pooler_output
        self.text_embeds = text_embeds
        self.image_embeds = image_embeds


def test_get_vector_uses_pooler_output_first():
    vector = torch.tensor([[3.0, 4.0]])
    output = FakeOutput(pooler_output=vector)

    assert get_vector(output) is vector


def test_get_vector_handles_text_and_image_embeds():
    text_vector = torch.tensor([[1.0, 0.0]])
    image_vector = torch.tensor([[0.0, 1.0]])

    assert get_vector(FakeOutput(text_embeds=text_vector)) is text_vector
    assert get_vector(FakeOutput(image_embeds=image_vector)) is image_vector


def test_normalize_returns_unit_vectors():
    vectors = normalize(torch.tensor([[3.0, 4.0], [0.0, 2.0]]))
    norms = vectors.norm(dim=1)

    assert torch.allclose(norms, torch.ones(2))
