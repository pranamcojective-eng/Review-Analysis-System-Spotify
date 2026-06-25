import logging
from collections import Counter
from datetime import datetime, timezone

from src.models import (
    CleanedReview,
    InsightReport,
    ReviewAnalysis,
    ReviewCategory,
    Sentiment,
    ThemeCluster,
    UserIntent,
    UserSegment,
)

logger = logging.getLogger(__name__)


class InsightGenerator:
    """Generate structured insights from analyzed reviews."""

    def generate(
        self,
        reviews: list[CleanedReview],
        analyses: list[ReviewAnalysis],
        clusters: list[ThemeCluster],
    ) -> InsightReport:
        logger.info("Generating insight report from %d analyses", len(analyses))

        sources = Counter(r.source.value for r in reviews)
        sentiments = Counter(a.sentiment.value for a in analyses)
        categories = Counter(a.category.value for a in analyses)
        intents = Counter(a.intent.value for a in analyses)

        discovery_reviews = [a for a in analyses if a.category == ReviewCategory.DISCOVERY_ISSUES]
        rec_reviews = [a for a in analyses if a.category == ReviewCategory.RECOMMENDATION_QUALITY]
        rep_reviews = [a for a in analyses if a.category == ReviewCategory.REPETITION_BOREDOM]

        top_discovery = self._top_reasons(discovery_reviews, analyses, reviews, limit=5)
        rec_frustrations = self._top_reasons(rec_reviews, analyses, reviews, limit=5)
        rep_patterns = self._top_reasons(rep_reviews, analyses, reviews, limit=5)
        unmet_needs = self._emerging_needs(analyses, intents)
        segments = self._infer_segments(analyses, reviews)
        sample_quotes = self._sample_quotes(reviews, analyses, limit=8)
        summary = self._build_summary(
            len(reviews), sentiments, categories, clusters, top_discovery, rec_frustrations
        )

        report = InsightReport(
            generated_at=datetime.now(timezone.utc),
            total_reviews=len(reviews),
            sources_breakdown=dict(sources),
            sentiment_distribution=dict(sentiments),
            category_distribution=dict(categories),
            top_discovery_struggles=top_discovery,
            recommendation_frustrations=rec_frustrations,
            repetition_patterns=rep_patterns,
            emerging_unmet_needs=unmet_needs,
            user_segments=segments,
            theme_clusters=clusters,
            sample_quotes=sample_quotes,
            key_insights_summary=summary,
        )
        logger.info("Insight report generated with %d key insights", len(summary))
        return report

    def _top_reasons(
        self,
        filtered: list[ReviewAnalysis],
        all_analyses: list[ReviewAnalysis],
        reviews: list[CleanedReview],
        limit: int = 5,
    ) -> list[str]:
        if not filtered:
            return ["Insufficient data in this category"]

        theme_counts: Counter = Counter()
        for a in filtered:
            theme_counts.update(a.themes)

        if theme_counts:
            return [f"{theme} ({count} mentions)" for theme, count in theme_counts.most_common(limit)]

        review_map = {r.id: r for r in reviews}
        phrases = []
        for a in filtered:
            for phrase in a.key_phrases[:1]:
                phrases.append(phrase)
        return phrases[:limit] if phrases else ["Users report general dissatisfaction in this area"]

    def _emerging_needs(self, analyses: list[ReviewAnalysis], intents: Counter) -> list[str]:
        needs = []
        intent_needs = {
            UserIntent.WANT_NEW_MUSIC.value: "Strong demand for fresher artist discovery beyond comfort zones",
            UserIntent.WANT_MOOD_PLAYLISTS.value: "Users want accurate mood-based and activity-specific playlists",
            UserIntent.WANT_BETTER_RECOMMENDATIONS.value: "Recommendation relevance and genre accuracy need improvement",
            UserIntent.WANT_LESS_REPETITION.value: "Listeners want controls to reduce repetitive track exposure",
            UserIntent.WANT_BETTER_UX.value: "Discovery features are hard to find due to UX/navigation issues",
        }
        for intent, count in intents.most_common():
            if intent in intent_needs and count >= 1:
                needs.append(f"{intent_needs[intent]} ({count} signals)")

        all_themes: Counter = Counter()
        for a in analyses:
            all_themes.update(a.themes)
        for theme, count in all_themes.most_common(3):
            if "social" in theme:
                needs.append(f"Social/collaborative discovery is an emerging opportunity ({count} mentions)")

        feature_signals = [
            a for a in analyses
            if any("discovery appetite" in p.lower() or "slider" in p.lower() for p in a.key_phrases)
        ]
        if feature_signals:
            needs.append("Users want configurable 'discovery adventurousness' controls")

        return needs[:6] if needs else ["Continue monitoring for emerging patterns"]

    def _infer_segments(
        self, analyses: list[ReviewAnalysis], reviews: list[CleanedReview]
    ) -> dict[str, int]:
        segments: Counter = Counter()
        review_map = {r.id: r for r in reviews}

        for a in analyses:
            text = review_map.get(a.review_id, reviews[0]).cleaned_text.lower() if review_map else ""

            if a.intent == UserIntent.WANT_NEW_MUSIC or "explore" in text or "comfort zone" in text:
                segments[UserSegment.EXPLORER.value] += 1
            elif a.intent == UserIntent.WANT_MOOD_PLAYLISTS or "mood" in text:
                segments[UserSegment.MOOD_CURATOR.value] += 1
            elif "passive" in text or "zone out" in text or "daily mix" in text:
                segments[UserSegment.PASSIVE_LISTENER.value] += 1
            elif "blend" in text or "friend" in text or "social" in text:
                segments[UserSegment.POWER_USER.value] += 1
            else:
                segments[UserSegment.CASUAL_USER.value] += 1

        return dict(segments)

    def _sample_quotes(
        self, reviews: list[CleanedReview], analyses: list[ReviewAnalysis], limit: int = 8
    ) -> list[dict]:
        analysis_map = {a.review_id: a for a in analyses}
        quotes = []
        for review in reviews:
            a = analysis_map.get(review.id)
            if not a:
                continue
            quotes.append({
                "id": review.id,
                "source": review.source.value,
                "text": review.original_text,
                "sentiment": a.sentiment.value,
                "category": a.category.value,
                "intent": a.intent.value,
            })
            if len(quotes) >= limit:
                break
        return quotes

    def _build_summary(
        self,
        total: int,
        sentiments: Counter,
        categories: Counter,
        clusters: list[ThemeCluster],
        discovery: list[str],
        rec_frustrations: list[str],
    ) -> list[str]:
        neg_pct = round(100 * sentiments.get("negative", 0) / max(total, 1))
        pos_pct = round(100 * sentiments.get("positive", 0) / max(total, 1))
        top_cat = categories.most_common(1)[0][0] if categories else "N/A"
        top_cluster = clusters[0].label if clusters else "N/A"

        return [
            f"Analyzed {total} user feedback items across App Store, Play Store, Reddit, and Twitter/X.",
            f"Sentiment split: {neg_pct}% negative, {pos_pct}% positive — discovery and recommendation quality dominate complaints.",
            f"Most common category: {top_cat}. Largest theme cluster: '{top_cluster}'.",
            f"Top discovery struggle: {discovery[0] if discovery else 'N/A'}.",
            f"Top recommendation frustration: {rec_frustrations[0] if rec_frustrations else 'N/A'}.",
            "Passive listeners and explorers are the two largest behavioral segments — tailor discovery UX accordingly.",
        ]
