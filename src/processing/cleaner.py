import logging
import re
import unicodedata

from src.models import CleanedReview, RawReview

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


class TextCleaner:
    """Clean and normalize review text."""

    def __init__(self, remove_emojis: bool = True, lowercase: bool = False):
        self.remove_emojis = remove_emojis
        self.lowercase = lowercase

    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = URL_PATTERN.sub("", text)
        text = MENTION_PATTERN.sub("", text)
        if self.remove_emojis:
            text = EMOJI_PATTERN.sub("", text)
        text = WHITESPACE_PATTERN.sub(" ", text).strip()
        if self.lowercase:
            text = text.lower()
        return text

    def process_batch(self, reviews: list[RawReview]) -> list[CleanedReview]:
        logger.info("Cleaning %d reviews", len(reviews))
        cleaned = []
        for review in reviews:
            cleaned_text = self.clean(review.text)
            if len(cleaned_text) < 5:
                logger.debug("Skipping review %s (too short after cleaning)", review.id)
                continue
            cleaned.append(
                CleanedReview(
                    id=review.id,
                    source=review.source,
                    original_text=review.text,
                    cleaned_text=cleaned_text,
                    rating=review.rating,
                    author=review.author,
                    timestamp=review.timestamp,
                    metadata=review.metadata,
                )
            )
        logger.info("Cleaned %d reviews (%d skipped)", len(cleaned), len(reviews) - len(cleaned))
        return cleaned
