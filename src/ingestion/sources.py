import logging
from typing import Callable

from src.models import RawReview

from .mock_loader import MockReviewLoader

logger = logging.getLogger(__name__)


class IngestionManager:
    """Orchestrates ingestion from multiple data sources."""

    def __init__(self):
        self._ingestors: list[tuple[str, Callable[[], list[RawReview]]]] = []

    def register(self, name: str, ingestor_fn: Callable[[], list[RawReview]]) -> None:
        self._ingestors.append((name, ingestor_fn))

    def ingest_all(self) -> list[RawReview]:
        all_reviews: list[RawReview] = []
        for name, fn in self._ingestors:
            logger.info("Running ingestor: %s", name)
            batch = fn()
            logger.info("  -> %d reviews from %s", len(batch), name)
            all_reviews.extend(batch)
        logger.info("Total ingested: %d reviews", len(all_reviews))
        return all_reviews


def create_default_ingestion_manager() -> IngestionManager:
    manager = IngestionManager()
    manager.register("mock_data", MockReviewLoader().ingest)
    return manager
