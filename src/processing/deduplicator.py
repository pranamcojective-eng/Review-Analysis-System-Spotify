import logging
from difflib import SequenceMatcher

from src.models import CleanedReview

logger = logging.getLogger(__name__)


class Deduplicator:
    """Remove duplicate and near-duplicate reviews."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def deduplicate(self, reviews: list[CleanedReview]) -> tuple[list[CleanedReview], int]:
        logger.info("Deduplicating %d reviews (threshold=%.2f)", len(reviews), self.similarity_threshold)
        unique: list[CleanedReview] = []
        removed = 0

        for review in reviews:
            is_dup = False
            for existing in unique:
                if self._similarity(review.cleaned_text, existing.cleaned_text) >= self.similarity_threshold:
                    is_dup = True
                    logger.debug("Duplicate: %s ~ %s", review.id, existing.id)
                    break
            if is_dup:
                removed += 1
            else:
                unique.append(review)

        logger.info("Removed %d duplicates, %d unique remain", removed, len(unique))
        return unique, removed
