def extract_entities(text: str):
    """Extract simple 'entities': numbers, capitalized words, dates."""
    numbers = re.findall(r'\b\d+(?:[,.\d]*)\b', text)
    caps = re.findall(r'\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b', text)
    dates = re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?\b', text, flags=re.I)
    return {"numbers": numbers, "caps": caps, "dates": dates}


def tfidf_similarity(a: str, b: str) -> float:
    """Semantic similarity (cosine) between two texts using TF-IDF."""
    if not a or not b:
        return 0.0
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
    X = vect.fit_transform([a, b])
    return float(cosine_similarity(X[0:1], X[1:2])[0][0])
