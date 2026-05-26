import json
from pathlib import Path
from eval.metrics import mrr, recall_at_k
from fashionrag.hybrid import hybrid_search
from fashionrag.qdrant import search_qdrant

TEST_FILE = Path("eval/test_queries.json")
TOP_K = 5

def load_test_queries(path: Path = TEST_FILE) -> list[dict]:
    with open(path, "r") as file:
        return json.load(file)

def score_search(name: str, search_func, test_queries: list[dict]) -> dict:
    recalls, mrrs = [], []

    for row in test_queries:
        results = search_func(row["query"], limit=TOP_K)
        recalls.append(recall_at_k(results, row["relevant_ids"], k=TOP_K))
        mrrs.append(mrr(results, row["relevant_ids"]))

    return {
        "name": name,
        "recall": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
    }


def main():
    test_queries = load_test_queries()
    dense = score_search("Dense only:", search_qdrant, test_queries)
    hybrid = score_search("Hybrid:", hybrid_search, test_queries)

    print("Method       Recall@5      MRR")
    print("---------------------------")

    for row in [dense, hybrid]:
        print(f"{row['name']:<12} {row['recall']:.3f}         {row['mrr']:.3f}")


if __name__ == "__main__":
    main()
