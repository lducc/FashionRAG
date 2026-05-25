from fashionrag.hybrid import hybrid_search


query = input("Enter a query: ")

print(f"\nSearch: {query}")

results = hybrid_search(query)

for product in results:
    print(f"{product['name']} | {product['color']} | "
            f"{product['gender']} | semantic={product['semantic_score']} | "
            f"keyword={product['keyword_score']} | final={product['final_score']}")
