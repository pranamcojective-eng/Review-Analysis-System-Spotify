import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from src.config import DATA_DIR
from src.models import DataSource, RawReview

logger = logging.getLogger(__name__)


class BaseIngestor(ABC):
    @abstractmethod
    def ingest(self) -> list[RawReview]:
        pass


class MockReviewLoader(BaseIngestor):
    """Load mock review data from JSON file."""

    def __init__(self, filepath: Path | None = None):
        self.filepath = filepath or DATA_DIR / "mock_reviews.json"

    def ingest(self) -> list[RawReview]:
        logger.info("Ingesting mock reviews from %s", self.filepath)
        with open(self.filepath, encoding="utf-8") as f:
            data = json.load(f)

        reviews = []
        for item in data:
            ts = item.get("timestamp")
            reviews.append(
                RawReview(
                    id=item["id"],
                    source=DataSource(item["source"]),
                    text=item["text"],
                    rating=item.get("rating"),
                    author=item.get("author"),
                    timestamp=datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None,
                    metadata=item.get("metadata", {}),
                )
            )
        logger.info("Loaded %d raw reviews", len(reviews))
        return reviews
