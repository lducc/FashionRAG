from nltk.tokenize import wordpunct_tokenize

from fashionrag.products import load_products


FILTER_FIELDS = ["gender", "article_type", "color", "usage", "season"]
EXACT_ARTICLE_TYPES = {"shirts", "tshirts"}
WORD_ALIASES = {
    "man": "men",
    "male": "men",
    "woman": "women",
    "female": "women",
    "lady": "women",
    "ladies": "women",
    "boy": "boys",
    "girl": "girls",
    "grey": "gray",
    "tee": "tshirt",
    "tees": "tshirt",
    "sari": "saree",
    "sneaker": "shoe",
    "sneakers": "shoe",
}
metadata_index = None


def query_words(query):
    words = []

    for word in wordpunct_tokenize(query.lower()):
        if word.isalnum():
            words.append(word)

    return words


def normalize_word(word):
    word = WORD_ALIASES.get(word.lower(), word.lower())

    if word == "shoes":
        return "shoe"

    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"

    if word.endswith("sses") and len(word) > 4:
        return word[:-2]

    if word.endswith("ss"):
        return word

    if word.endswith("es") and len(word) > 3:
        return word[:-2]

    if word.endswith("s") and len(word) > 3:
        return word[:-1]

    return word


def normalize_text(text):
    words = []

    for word in wordpunct_tokenize(str(text).lower()):
        if word.isalnum():
            words.append(normalize_word(word))

    return words


def has_tshirt(words):
    if "tshirt" in words:
        return True

    for index, word in enumerate(words[:-1]):
        if word == "t" and words[index + 1] == "shirt":
            return True

    return False


def build_metadata_index(products):
    index = {}

    for field in FILTER_FIELDS:
        index[field] = []
        seen_values = set()

        for product in products:
            value = product[field]

            if value in seen_values:
                continue

            seen_values.add(value)
            index[field].append(
                {
                    "value": value,
                    "tokens": set(normalize_text(value)),
                }
            )

    return index


def load_metadata_index():
    global metadata_index

    if metadata_index is None:
        metadata_index = build_metadata_index(load_products())

    return metadata_index


def find_filter_value(words, field, index):
    word_set = set(words)
    best_value = None
    best_size = 0

    for row in index[field]:
        tokens = row["tokens"]

        if tokens and tokens.issubset(word_set) and len(tokens) > best_size:
            best_value = row["value"]
            best_size = len(tokens)

    return best_value


def extract_filters(query):
    words = [normalize_word(word) for word in query_words(query)]
    index = load_metadata_index()
    filters = {}

    for field in FILTER_FIELDS:
        value = find_filter_value(words, field, index)

        if value:
            filters[field] = value

    if has_tshirt(words):
        filters["article_type"] = "Tshirts"

    return filters


def product_matches(product, filters):
    for field, expected in filters.items():
        actual = str(product.get(field, "")).lower()
        expected = expected.lower()

        if field == "gender":
            if actual != expected:
                return False
        elif field == "article_type" and expected in EXACT_ARTICLE_TYPES:
            if actual != expected:
                return False
        elif expected not in actual and actual not in expected:
            return False

    return True


def filter_products(products, filters):
    if not filters:
        return products

    filtered_products = []

    for product in products:
        if product_matches(product, filters):
            filtered_products.append(product)

    return filtered_products
