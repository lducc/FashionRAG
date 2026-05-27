from nltk.tokenize import wordpunct_tokenize


GENDER_WORDS = {
    "man": "Men",
    "men": "Men",
    "male": "Men",
    "woman": "Women",
    "women": "Women",
    "female": "Women",
    "lady": "Women",
    "ladies": "Women",
    "boy": "Boys",
    "boys": "Boys",
    "girl": "Girls",
    "girls": "Girls",
    "unisex": "Unisex",
}

ARTICLE_WORDS = {
    "dress": "Dresses",
    "dresses": "Dresses",
    "jean": "Jeans",
    "jeans": "Jeans",
    "kurta": "Kurtas",
    "saree": "Sarees",
    "sari": "Sarees",
    "top": "Tops",
    "tops": "Tops",
    "watch": "Watches",
    "watches": "Watches",
    "heel": "Heels",
    "heels": "Heels",
    "sandal": "Sandals",
    "sandals": "Sandals",
    "lipstick": "Lipstick",
    "handbag": "Handbags",
    "bag": "Bags",
    "bags": "Bags",
    "shoe": "Shoes",
    "shoes": "Shoes",
    "sneaker": "Shoes",
    "sneakers": "Shoes",
}

COLOR_WORDS = {
    "black": "Black",
    "blue": "Blue",
    "brown": "Brown",
    "green": "Green",
    "grey": "Grey",
    "gray": "Grey",
    "maroon": "Maroon",
    "navy": "Navy Blue",
    "orange": "Orange",
    "pink": "Pink",
    "purple": "Purple",
    "red": "Red",
    "silver": "Silver",
    "white": "White",
    "yellow": "Yellow",
}

USAGE_WORDS = {
    "casual": "Casual",
    "formal": "Formal",
    "party": "Party",
    "sports": "Sports",
    "sport": "Sports",
    "travel": "Travel",
}

SEASON_WORDS = {
    "fall": "Fall",
    "spring": "Spring",
    "summer": "Summer",
    "winter": "Winter",
}

EXACT_ARTICLE_TYPES = {"shirts", "tshirts"}


def query_words(query):
    words = []

    for word in wordpunct_tokenize(query.lower()):
        if word.isalnum():
            words.append(word)

    return words


def has_tshirt(words):
    if "tshirt" in words or "tee" in words or "tees" in words:
        return True

    for index, word in enumerate(words[:-1]):
        if word == "t" and words[index + 1] == "shirt":
            return True

    return False


def extract_filters(query):
    words = query_words(query)
    word_set = set(words)
    filters = {}

    for word, gender in GENDER_WORDS.items():
        if word in word_set:
            filters["gender"] = gender
            break

    if has_tshirt(words):
        filters["article_type"] = "Tshirts"
    elif "shirt" in word_set or "shirts" in word_set:
        filters["article_type"] = "Shirts"
    else:
        for word, article_type in ARTICLE_WORDS.items():
            if word in word_set:
                filters["article_type"] = article_type
                break

    for word, color in COLOR_WORDS.items():
        if word in word_set:
            filters["color"] = color
            break

    for word, usage in USAGE_WORDS.items():
        if word in word_set:
            filters["usage"] = usage
            break

    for word, season in SEASON_WORDS.items():
        if word in word_set:
            filters["season"] = season
            break

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
