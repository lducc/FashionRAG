from fashionrag.qdrant import upload_products
from fashionrag.settings import QDRANT_COLLECTION


count = upload_products()
print(f"Uploaded {count} products to {QDRANT_COLLECTION}")
