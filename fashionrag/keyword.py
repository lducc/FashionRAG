from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from rank_bm25 import BM25Okapi
import nltk
from fashionrag.products import load_products

nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def tokenize(text):
    words = wordpunct_tokenize(text.lower())
    tokens = []

    for word in words:
        if not word.isalnum() or word in stop_words:
            continue

        word = stemmer.stem(word)
        tokens.append(word)

        if word == "shirt":
            tokens.append("tshirt")
        elif word == "tshirt":
            tokens.append("t")
            tokens.append("shirt")
        elif word == "tee":
            tokens.append("tshirt")

    return tokens


def keyword_search(query, limit=30):
    products = load_products()
    tokenized_products = []

    for product in products:
        tokenized_products.append(tokenize(product["search_text"]))

    bm25 = BM25Okapi(tokenized_products)
    query_words = tokenize(query)
    scores = bm25.get_scores(query_words)
    ranked_indexes = scores.argsort()[::-1]

    seen_names = []
    results = []

    for index in ranked_indexes:
        product = products[index]

        if product["name"] in seen_names:
            continue

        seen_names.append(product["name"])
        product["score"] = round(float(scores[index]), 3)
        results.append(product)

        if len(results) == limit:
            break

    return results
