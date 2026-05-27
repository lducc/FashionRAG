from fashionrag import hybrid


def test_hybrid_search_ignores_zero_scores(monkeypatch):
    semantic_results = [
        {"id": "1", "name": "semantic hit", "score": 0.8},
        {"id": "2", "name": "zero semantic", "score": 0.0},
    ]
    keyword_results = [
        {"id": "3", "name": "keyword hit", "score": 9.5},
        {"id": "4", "name": "zero keyword", "score": 0.0},
    ]

    monkeypatch.setattr(hybrid, "search_qdrant", lambda query, limit=30: semantic_results)
    monkeypatch.setattr(hybrid, "keyword_search", lambda query, limit=30: keyword_results)

    results = hybrid.hybrid_search("red shirt", limit=10)
    ids = [product["id"] for product in results]

    assert ids == ["1", "3"]
    assert results[0]["semantic_score"] == 0.8
    assert results[0]["keyword_score"] == 0
    assert results[0]["final_score"] > 0


def test_hybrid_search_keeps_both_scores_for_same_product(monkeypatch):
    semantic_results = [{"id": "1", "name": "same product", "score": 0.8}]
    keyword_results = [{"id": "1", "name": "same product", "score": 5.0}]

    monkeypatch.setattr(hybrid, "search_qdrant", lambda query, limit=30: semantic_results)
    monkeypatch.setattr(hybrid, "keyword_search", lambda query, limit=30: keyword_results)

    results = hybrid.hybrid_search("red shirt", limit=10)

    assert len(results) == 1
    assert results[0]["semantic_score"] == 0.8
    assert results[0]["keyword_score"] == 5.0
    assert results[0]["final_score"] == 0.0328


def test_hybrid_search_applies_metadata_filters(monkeypatch):
    semantic_results = [
        {
            "id": "1",
            "name": "wrong gender",
            "score": 0.9,
            "gender": "Men",
            "article_type": "Dresses",
            "color": "Black",
        },
        {
            "id": "2",
            "name": "right product",
            "score": 0.8,
            "gender": "Women",
            "article_type": "Dresses",
            "color": "Black",
        },
    ]
    keyword_results = []

    monkeypatch.setattr(hybrid, "search_qdrant", lambda query, limit=50: semantic_results)
    monkeypatch.setattr(hybrid, "keyword_search", lambda query, limit=50: keyword_results)

    results = hybrid.hybrid_search("women black dress", limit=7)

    assert [product["id"] for product in results] == ["2"]


def test_hybrid_search_falls_back_when_filters_remove_everything(monkeypatch):
    semantic_results = [
        {
            "id": "1",
            "name": "best fallback",
            "score": 0.9,
            "gender": "Men",
            "article_type": "Shirts",
            "color": "Red",
        }
    ]
    keyword_results = []

    monkeypatch.setattr(hybrid, "search_qdrant", lambda query, limit=50: semantic_results)
    monkeypatch.setattr(hybrid, "keyword_search", lambda query, limit=50: keyword_results)

    results = hybrid.hybrid_search("women black dress", limit=7)

    assert [product["id"] for product in results] == ["1"]
