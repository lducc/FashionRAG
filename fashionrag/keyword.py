import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from rank_bm25 import BM25Okapi
from fashionrag.products import load_products
from fashionrag.settings import BM25_FILE

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
#Caching bm25 + products
bm25, products = None, None

def build_keyword_index(product_list):
    tokenized_products = []

    for product in product_list:
        tokenized_products.append(tokenize(product["search_text"]))

    return BM25Okapi(tokenized_products)


def save_keyword_index(product_list, path=BM25_FILE):
    path.parent.mkdir(exist_ok=True)
    keyword_index = build_keyword_index(product_list)

    with path.open("wb") as file:
        pickle.dump(keyword_index, file)

    return keyword_index

def load_keyword_index():
    global bm25, products

    if products is None:
        products = load_products()

    if bm25 is None:
        if not BM25_FILE.exists():
            raise FileNotFoundError(f"{BM25_FILE} is missing. Run ingestion first.")

        with BM25_FILE.open("rb") as file:
            bm25 = pickle.load(file)

    return products, bm25

def tokenize(text):
    words = wordpunct_tokenize(text.lower())
    tokens = []

    for word in words:
        if not word.isalnum() or word in stop_words:
            continue

        word = stemmer.stem(word)
        tokens.append(word)

        #specific case for t-shirts?
        #TASK: add metadata in the dataset for other clothings as well
        if word == "shirt":
            tokens.append("tshirt")
        elif word == "tshirt":
            tokens.append("t")
            tokens.append("shirt")
        elif word == "tee":
            tokens.append("tshirt")

    return tokens


def keyword_search(query, limit=30):
    products, bm25 = load_keyword_index()
    query_words = tokenize(query)
    scores = bm25.get_scores(query_words)
    ranked_indexes = scores.argsort()[::-1]

    seen_names = []
    results = []

    for index in ranked_indexes:
        product = products[index]
        score = float(scores[index])

        if score <= 0:
            break

        if product["name"] in seen_names:
            continue

        seen_names.append(product["name"])
        product["score"] = round(score, 3)
        results.append(product)

        if len(results) == limit:
            break

    return results
