from fashionrag.filters import extract_filters, filter_products
from fashionrag.keyword import keyword_search
from fashionrag.qdrant import search_qdrant


def hybrid_search(query, limit=7):
    filters = extract_filters(query)
    search_limit = 50 if filters else 30
    semantic_results = search_qdrant(query, limit=search_limit)
    keyword_results = keyword_search(query, limit=search_limit)

    final_scores = {}
    semantic_scores = {}
    keyword_scores = {}
    products = {}
    k = 60

    for rank, product in enumerate(semantic_results):
        product_id = product["id"]

        if product["score"] <= 0:
            continue

        final_scores[product_id] = final_scores.get(product_id, 0) + 1 / (k + rank + 1)
        semantic_scores[product_id] = product["score"]
        products[product_id] = product

    for rank, product in enumerate(keyword_results):
        product_id = product["id"]

        if product["score"] <= 0:
            continue

        final_scores[product_id] = final_scores.get(product_id, 0) + 1 / (k + rank + 1)
        keyword_scores[product_id] = product["score"]
        products[product_id] = product

    ranked_ids = sorted(final_scores, key=final_scores.get, reverse=True)
    results = []

    for product_id in ranked_ids:
        product = products[product_id]
        product["semantic_score"] = semantic_scores.get(product_id, 0)
        product["keyword_score"] = keyword_scores.get(product_id, 0)
        product["final_score"] = round(final_scores[product_id], 4)
        results.append(product)

    filtered_results = filter_products(results, filters)

    if filtered_results:
        return filtered_results[:limit]

    return results[:limit]
