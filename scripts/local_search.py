from fashionrag.search import local_search


query = input("Enter a query: ")

print(f"\nSearch: {query}")

results = local_search(query)

for product in results:
    print(f"{product['name']} | {product['color']} | "
            f"{product['gender']} | score={product['score']}")
