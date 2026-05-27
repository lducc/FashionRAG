from fashionrag.hybrid import hybrid_search
from fashionrag.keyword import keyword_search
from fashionrag.qdrant import search_qdrant
from fashionrag.search import local_search


SEARCH_MODES = {
    "hybrid": hybrid_search,
    "semantic": search_qdrant,
    "keyword": keyword_search,
    "local": local_search,
}


def print_result(product, mode):
    if mode == "hybrid":
        print(f"{product['name']} | {product['color']} | "
              f"{product['gender']} | semantic={product['semantic_score']} | "
              f"keyword={product['keyword_score']} | final={product['final_score']}")
    else:
        print(f"{product['name']} | {product['color']} | "
              f"{product['gender']} | score={product['score']}")


def main():
    print("Modes: hybrid, semantic, keyword, local")
    mode = input("Choose mode: ").strip().lower()

    if mode not in SEARCH_MODES:
        print(f"Unknown mode: {mode}")
        return

    query = input("Enter a query: ")
    print(f"\nSearch: {query}")

    results = SEARCH_MODES[mode](query)

    for product in results:
        print_result(product, mode)


if __name__ == "__main__":
    main()
