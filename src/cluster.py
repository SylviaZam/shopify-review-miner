from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def cluster_comments(comments: list[str], k: int = 8):
    """Return (labels, kmeans_model)."""
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(comments)
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    return labels, km
