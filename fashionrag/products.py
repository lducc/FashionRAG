import json
from fashionrag.settings import PRODUCTS_FILE

def build_product(row):
    product = {
        "id": str(row["id"]),
        "name": row["productDisplayName"],
        "category": row["masterCategory"],
        "subcategory": row["subCategory"],
        "article_type": row["articleType"],
        "color": row["baseColour"],
        "season": row["season"],
        "gender": row["gender"],
        "usage": row["usage"],
        "year": int(row["year"]),
    }

    product["search_text"] = (product["name"] + " "
                            + product["category"] + " "
                            + product["subcategory"] + " "
                            + product["article_type"] + " "
                            + product["color"] + " "
                            + product["season"] + " "
                            + product["gender"] + " "
                            + product["usage"]).lower()

    return product


def load_products(path=PRODUCTS_FILE):
    products = []

    with path.open("r") as file:
        for line in file:
            products.append(json.loads(line))

    return products


def save_products(products, path=PRODUCTS_FILE):
    path.parent.mkdir(exist_ok=True)

    with path.open("w") as file:
        for product in products:
            file.write(json.dumps(product) + "\n")
