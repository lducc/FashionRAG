from fashionrag.keyword import keyword_search

query = input("Enter a query: ")
print(f"\nSearch: {query}")

results = keyword_search(query, limit=7)

for product in results:
    print(f"{product['name']} | {product['color']} | "
            f"{product['gender']} | score={product['score']}")
