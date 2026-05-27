from fashionrag import keyword
from fashionrag.products import build_product


def test_tokenize_preprocesses_query_text():
    tokens = keyword.tokenize("Striped red and blue T-shirt tee shirts")

    assert "and" not in tokens
    assert "stripe" in tokens
    assert "red" in tokens
    assert "blue" in tokens
    assert "tshirt" in tokens
    assert "shirt" in tokens


def test_build_keyword_index_uses_tokenize(monkeypatch):
    calls = []

    def fake_tokenize(text):
        calls.append(text)
        return ["shirt"]

    monkeypatch.setattr(keyword, "tokenize", fake_tokenize)
    keyword.build_keyword_index([{"search_text": "red shirt"}])

    assert calls == ["red shirt"]


def test_product_search_text_includes_metadata():
    row = {
        "id": 1,
        "productDisplayName": "Blue Shirt",
        "masterCategory": "Apparel",
        "subCategory": "Topwear",
        "articleType": "Shirts",
        "baseColour": "Blue",
        "season": "Summer",
        "gender": "Men",
        "usage": "Casual",
        "year": 2012,
    }

    product = build_product(row)

    assert product["search_text"] == (
        "blue shirt apparel topwear shirts blue summer men casual"
    )
