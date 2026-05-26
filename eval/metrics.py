def recall_at_k(results: list[dict], relevant_ids: list[str], k: int = 5) -> float:
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return 0.0

    retrieved_ids = set()

    for product in results[:k]:
        retrieved_ids.add(str(product["id"]))

    return len(retrieved_ids & relevant_ids) / len(relevant_ids)


def mrr(results: list[dict], relevant_ids: list[str]) -> float:
    relevant_ids = set(relevant_ids)

    for rank, product in enumerate(results, start=1):
        if str(product["id"]) in relevant_ids:
            return 1 / rank

    return 0.0
