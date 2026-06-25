"""Data ingestion layer - load reviews from multiple sources."""

from .mock_loader import MockReviewLoader
from .sources import IngestionManager

__all__ = ["IngestionManager", "MockReviewLoader"]
