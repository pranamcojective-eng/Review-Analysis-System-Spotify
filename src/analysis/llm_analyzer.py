import json
import logging
import re
from typing import Optional

from src.config import settings
from src.models import (
    CleanedReview,
    ReviewAnalysis,
    ReviewCategory,
    Sentiment,
    UserIntent,
)

logger = logging.getLogger(__name__)

CATEGORY_KEYWORDS = {
    ReviewCategory.DISCOVERY_ISSUES: [
        "discover", "new music", "new artist", "find music", "discovery",
        "explore", "comfort zone", "fresh music",
    ],
    ReviewCategory.RECOMMENDATION_QUALITY: [
        "recommend", "algorithm", "suggestion", "daily mix", "discover weekly",
        "radio", "release radar", "wrong genre", "broken",
    ],
    ReviewCategory.REPETITION_BOREDOM: [
        "same song", "repeat", "over and over", "recycled", "bored",
        "same track", "same artist", "nostalgia", "stuck",
    ],
    ReviewCategory.UX_ISSUES: [
        "ui", "ux", "navigation", "cluttered", "confusing", "search",
        "hard to find", "taps", "interface",
    ],
    ReviewCategory.POSITIVE_FEEDBACK: [
        "love", "best", "amazing", "fire", "great", "shoutout",
        "game changer", "keep it up", "introduced me",
    ],
}

INTENT_KEYWORDS = {
    UserIntent.WANT_NEW_MUSIC: ["new music", "new artist", "fresh", "discover", "outside comfort"],
    UserIntent.WANT_MOOD_PLAYLISTS: ["mood", "chill", "workout", "sad", "vibes", "activity"],
    UserIntent.WANT_BETTER_RECOMMENDATIONS: ["recommend", "algorithm", "suggestion", "better"],
    UserIntent.WANT_LESS_REPETITION: ["same", "repeat", "bored", "recycled", "over and over"],
    UserIntent.WANT_BETTER_UX: ["ui", "ux", "navigation", "confusing", "cluttered", "search"],
    UserIntent.EXPRESSING_SATISFACTION: ["love", "best", "amazing", "fire", "great", "shoutout"],
}


class RuleBasedAnalyzer:
    """Fallback analyzer when no LLM API key is available."""

    def analyze(self, review: CleanedReview) -> ReviewAnalysis:
        text = review.cleaned_text.lower()

        category_scores = {}
        for cat, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                category_scores[cat] = score

        category = max(category_scores, key=category_scores.get) if category_scores else ReviewCategory.OTHER

        positive_words = ["love", "best", "amazing", "fire", "great", "shoutout", "game changer"]
        negative_words = ["terrible", "broken", "impossible", "bored", "mess", "confusing", "stuck", "worse"]
        pos = sum(1 for w in positive_words if w in text)
        neg = sum(1 for w in negative_words if w in text)

        if review.rating is not None:
            if review.rating >= 4:
                sentiment = Sentiment.POSITIVE
            elif review.rating <= 2:
                sentiment = Sentiment.NEGATIVE
            else:
                sentiment = Sentiment.NEUTRAL
        elif pos > neg:
            sentiment = Sentiment.POSITIVE
        elif neg > pos:
            sentiment = Sentiment.NEGATIVE
        else:
            sentiment = Sentiment.NEUTRAL

        intent_scores = {}
        for intent, keywords in INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                intent_scores[intent] = score
        intent = max(intent_scores, key=intent_scores.get) if intent_scores else UserIntent.OTHER

        themes = self._extract_themes(text)
        key_phrases = self._extract_key_phrases(review.cleaned_text)

        return ReviewAnalysis(
            review_id=review.id,
            category=category,
            sentiment=sentiment,
            intent=intent,
            themes=themes,
            key_phrases=key_phrases,
            confidence=0.65,
        )

    def _extract_themes(self, text: str) -> list[str]:
        themes = []
        theme_map = {
            "algorithm quality": ["algorithm", "recommend", "suggestion"],
            "repetitive playlists": ["same", "repeat", "recycled"],
            "discovery fatigue": ["discover", "new music", "comfort zone"],
            "mood matching": ["mood", "chill", "vibes"],
            "navigation UX": ["navigation", "ui", "ux", "search", "confusing"],
            "social discovery": ["friend", "blend", "social"],
        }
        for theme, keywords in theme_map.items():
            if any(kw in text for kw in keywords):
                themes.append(theme)
        return themes[:5]

    def _extract_key_phrases(self, text: str) -> list[str]:
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10][:3]


class LLMAnalyzer:
    """Analyze reviews using OpenAI LLM with rule-based fallback."""

    def __init__(self):
        self._rule_analyzer = RuleBasedAnalyzer()
        self._client: Optional[object] = None
        if settings.openai_api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=settings.openai_api_key)
                logger.info("LLM analyzer initialized with OpenAI (%s)", settings.openai_model)
            except Exception as e:
                logger.warning("Failed to init OpenAI client: %s. Using rule-based fallback.", e)
        else:
            logger.info("No OPENAI_API_KEY set. Using rule-based fallback analyzer.")

    def analyze_batch(self, reviews: list[CleanedReview]) -> list[ReviewAnalysis]:
        logger.info("Analyzing %d reviews", len(reviews))
        results = []
        for i, review in enumerate(reviews):
            if self._client:
                try:
                    analysis = self._analyze_with_llm(review)
                except Exception as e:
                    logger.warning("LLM failed for %s: %s. Using fallback.", review.id, e)
                    analysis = self._rule_analyzer.analyze(review)
            else:
                analysis = self._rule_analyzer.analyze(review)
            results.append(analysis)
            if (i + 1) % 5 == 0:
                logger.info("  Analyzed %d/%d reviews", i + 1, len(reviews))
        logger.info("Analysis complete for %d reviews", len(results))
        return results

    def _analyze_with_llm(self, review: CleanedReview) -> ReviewAnalysis:
        prompt = f"""Analyze this music streaming app user feedback.

Review: "{review.cleaned_text}"
Source: {review.source.value}
Rating: {review.rating or "N/A"}

Return JSON with exactly these fields:
- category: one of ["Discovery Issues", "Recommendation Quality", "Repetition / Boredom", "UX Issues", "Positive Feedback", "Other"]
- sentiment: one of ["positive", "negative", "neutral"]
- intent: one of ["want new music", "want mood-based playlists", "want better recommendations", "want less repetition", "want better UX", "expressing satisfaction", "other"]
- themes: array of 1-3 short theme strings
- key_phrases: array of 1-2 notable phrases from the review
- confidence: float 0-1

Return ONLY valid JSON, no markdown."""

        response = self._client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        data = json.loads(content)
        return ReviewAnalysis(
            review_id=review.id,
            category=ReviewCategory(data["category"]),
            sentiment=Sentiment(data["sentiment"]),
            intent=UserIntent(data["intent"]),
            themes=data.get("themes", []),
            key_phrases=data.get("key_phrases", []),
            confidence=float(data.get("confidence", 0.85)),
        )
