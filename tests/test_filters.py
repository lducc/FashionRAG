from fashionrag import filters as filters_module
from fashionrag.filters import build_metadata_index, extract_filters, filter_products


def test_extract_filters_finds_basic_metadata():
    filters = extract_filters("women black dress")

    assert filters == {
        "gender": "Women",
        "article_type": "Dresses",
        "color": "Black",
    }


def test_extract_filters_keeps_tshirt_and_shirt_separate():
    assert extract_filters("red t-shirt men")["article_type"] == "Tshirts"
    assert extract_filters("red tee men")["article_type"] == "Tshirts"
    assert extract_filters("red shirt men")["article_type"] == "Shirts"


def test_extract_filters_does_not_match_men_inside_women():
    assert extract_filters("women shoes")["gender"] == "Women"
    assert extract_filters("men shoes")["gender"] == "Men"


def test_filter_products_uses_exact_gender():
    products = [
        {"name": "men shirt", "gender": "Men", "article_type": "Shirts", "color": "Red"},
        {"name": "women shirt", "gender": "Women", "article_type": "Shirts", "color": "Red"},
    ]

    results = filter_products(products, {"gender": "Men"})

    assert results == [products[0]]


def test_filter_products_allows_broad_article_type():
    products = [
        {"name": "formal shoes", "article_type": "Formal Shoes", "gender": "Men"},
        {"name": "watch", "article_type": "Watches", "gender": "Men"},
    ]

    results = filter_products(products, {"article_type": "Shoes"})

    assert results == [products[0]]


def test_filter_products_keeps_shirts_and_tshirts_separate():
    products = [
        {"name": "plain shirt", "article_type": "Shirts", "gender": "Men"},
        {"name": "plain tshirt", "article_type": "Tshirts", "gender": "Men"},
    ]

    shirt_results = filter_products(products, {"article_type": "Shirts"})
    tshirt_results = filter_products(products, {"article_type": "Tshirts"})

    assert shirt_results == [products[0]]
    assert tshirt_results == [products[1]]


def test_extract_filters_uses_metadata_values(monkeypatch):
    products = [
        {
            "gender": "Women",
            "article_type": "Kurta Sets",
            "color": "Mustard",
            "usage": "Ethnic",
            "season": "Fall",
        }
    ]

    monkeypatch.setattr(filters_module, "metadata_index", build_metadata_index(products))

    assert extract_filters("women mustard kurta set") == {
        "gender": "Women",
        "article_type": "Kurta Sets",
        "color": "Mustard",
    }
