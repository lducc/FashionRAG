from fashionrag.qdrant import search_qdrant


query = input("Enter a query: ")

print(f"\nSearch: {query}")

results = search_qdrant(query)

for product in results:
    print(f"{product['name']} | {product['color']} | "
            f"{product['gender']} | score={product['score']}")
