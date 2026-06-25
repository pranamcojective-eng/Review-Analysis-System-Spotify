import logging
from collections import Counter
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from src.config import settings
from src.models import (
    CleanedReview,
    ReviewAnalysis,
    ReviewCategory,
    Sentiment,
    ThemeCluster,
)

from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)

CLUSTER_LABELS = {
    0: "Repetitive Recommendations",
    1: "Discovery & New Music",
    2: "Algorithm Quality Issues",
    3: "Mood & Playlist Needs",
    4: "UX & Navigation",
    5: "Positive Experiences",
}


class ThemeClusterer:
    """Cluster similar reviews using embeddings + KMeans."""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self._cluster_labels: dict[int, int] = {}

    def cluster(
        self,
        reviews: list[CleanedReview],
        analyses: list[ReviewAnalysis],
        n_clusters: Optional[int] = None,
    ) -> tuple[list[ThemeCluster], np.ndarray]:
        n = len(reviews)
        if n < 2:
            logger.warning("Too few reviews to cluster")
            return [], np.array([])

        n_clusters = min(n_clusters or settings.cluster_count, n)
        logger.info("Clustering %d reviews into %d clusters", n, n_clusters)

        texts = [r.cleaned_text for r in reviews]
        embeddings = self.embedding_service.encode(texts)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        self._cluster_labels = {reviews[i].id: int(labels[i]) for i in range(n)}

        analysis_map = {a.review_id: a for a in analyses}
        clusters = []

        for cluster_id in range(n_clusters):
            indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            if not indices:
                continue

            cluster_reviews = [reviews[i] for i in indices]
            cluster_analyses = [analysis_map[r.id] for r in cluster_reviews if r.id in analysis_map]

            all_themes = []
            for a in cluster_analyses:
                all_themes.extend(a.themes)
            theme_counts = Counter(all_themes)
            top_themes = [t for t, _ in theme_counts.most_common(5)]

            categories = [a.category for a in cluster_analyses]
            sentiments = [a.sentiment for a in cluster_analyses]
            dominant_category = Counter(categories).most_common(1)[0][0] if categories else ReviewCategory.OTHER
            dominant_sentiment = Counter(sentiments).most_common(1)[0][0] if sentiments else Sentiment.NEUTRAL

            label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
            if top_themes:
                label = top_themes[0].title()

            samples = [r.original_text[:120] + ("..." if len(r.original_text) > 120 else "") for r in cluster_reviews[:3]]

            clusters.append(
                ThemeCluster(
                    cluster_id=cluster_id,
                    label=label,
                    theme_keywords=top_themes or [label.lower()],
                    review_count=len(cluster_reviews),
                    sample_quotes=samples,
                    dominant_category=dominant_category,
                    dominant_sentiment=dominant_sentiment,
                )
            )

        clusters.sort(key=lambda c: c.review_count, reverse=True)
        logger.info("Created %d theme clusters", len(clusters))
        return clusters, embeddings

    def get_cluster_for_review(self, review_id: str) -> Optional[int]:
        return self._cluster_labels.get(review_id)
