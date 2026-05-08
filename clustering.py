"""clustering.py — K-Means customer segmentation"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

SEGMENT_NAMES = {
    0: "💎 Loyal Champions",
    1: "🧐 Price Sensitives",
    2: "⚠️ At-Risk Customers",
    3: "🌱 Occasional Buyers",
}


def find_optimal_k(X, k_range=range(2, 7)):
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) > 1:
            scores[k] = silhouette_score(X, labels)
    return max(scores, key=scores.get), scores


def run_clustering(X, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


def label_segments(labels):
    """Map numeric cluster → human-readable segment name."""
    return [SEGMENT_NAMES.get(l, f"Segment {l}") for l in labels]
