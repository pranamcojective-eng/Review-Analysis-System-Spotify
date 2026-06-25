from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DataSource(str, Enum):
    APP_STORE = "app_store"
    PLAY_STORE = "play_store"
    REDDIT = "reddit"
    TWITTER = "twitter"


class ReviewCategory(str, Enum):
    DISCOVERY_ISSUES = "Discovery Issues"
    RECOMMENDATION_QUALITY = "Recommendation Quality"
    REPETITION_BOREDOM = "Repetition / Boredom"
    UX_ISSUES = "UX Issues"
    POSITIVE_FEEDBACK = "Positive Feedback"
    OTHER = "Other"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class UserIntent(str, Enum):
    WANT_NEW_MUSIC = "want new music"
    WANT_MOOD_PLAYLISTS = "want mood-based playlists"
    WANT_BETTER_RECOMMENDATIONS = "want better recommendations"
    WANT_LESS_REPETITION = "want less repetition"
    WANT_BETTER_UX = "want better UX"
    EXPRESSING_SATISFACTION = "expressing satisfaction"
    OTHER = "other"


class UserSegment(str, Enum):
    PASSIVE_LISTENER = "passive listener"
    EXPLORER = "explorer"
    MOOD_CURATOR = "mood curator"
    POWER_USER = "power user"
    CASUAL_USER = "casual user"


class RawReview(BaseModel):
    id: str
    source: DataSource
    text: str
    rating: Optional[float] = None
    author: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)


class CleanedReview(BaseModel):
    id: str
    source: DataSource
    original_text: str
    cleaned_text: str
    rating: Optional[float] = None
    author: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)


class ReviewAnalysis(BaseModel):
    review_id: str
    category: ReviewCategory
    sentiment: Sentiment
    intent: UserIntent
    themes: list[str] = Field(default_factory=list)
    key_phrases: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ThemeCluster(BaseModel):
    cluster_id: int
    label: str
    theme_keywords: list[str]
    review_count: int
    sample_quotes: list[str]
    dominant_category: ReviewCategory
    dominant_sentiment: Sentiment


class InsightReport(BaseModel):
    generated_at: datetime
    total_reviews: int
    sources_breakdown: dict[str, int]
    sentiment_distribution: dict[str, int]
    category_distribution: dict[str, int]
    top_discovery_struggles: list[str]
    recommendation_frustrations: list[str]
    repetition_patterns: list[str]
    emerging_unmet_needs: list[str]
    user_segments: dict[str, int]
    theme_clusters: list[ThemeCluster]
    sample_quotes: list[dict]
    key_insights_summary: list[str]


class PipelineResult(BaseModel):
    raw_count: int
    cleaned_count: int
    duplicates_removed: int
    analyses: list[ReviewAnalysis]
    clusters: list[ThemeCluster]
    report: InsightReport
    log_entries: list[str]


class ChatSource(BaseModel):
    review_id: str
    source: str
    text: str
    score: float
    category: Optional[str] = None
    sentiment: Optional[str] = None
    intent: Optional[str] = None
    themes: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[ChatSource] = Field(default_factory=list)
