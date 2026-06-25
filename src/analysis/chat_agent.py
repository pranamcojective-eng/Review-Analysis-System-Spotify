"""RAG chat agent for answering questions from the review dataset."""

import logging
from collections import Counter
from typing import Optional

from src.analysis.embeddings import EmbeddingService
from src.analysis.vector_store import FAISSVectorStore
from src.config import settings
from src.models import ChatResponse, ChatSource, CleanedReview, InsightReport, ReviewAnalysis

logger = logging.getLogger(__name__)


class ReviewChatAgent:
    """Answer user questions using retrieved reviews and optional LLM synthesis."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
        reviews: list[CleanedReview],
        analyses: list[ReviewAnalysis],
        report: Optional[InsightReport] = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.review_map = {r.id: r for r in reviews}
        self.analysis_map = {a.review_id: a for a in analyses}
        self.report = report
        self._client: Optional[object] = None
        if settings.openai_api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=settings.openai_api_key)
            except Exception as e:
                logger.warning("Failed to init OpenAI for chat: %s", e)

    def ask(self, question: str, top_k: int = 5) -> ChatResponse:
        question = question.strip()
        if not question:
            return ChatResponse(
                question=question,
                answer="Please type a question about the review dataset.",
                sources=[],
            )

        embedding = self.embedding_service.encode([question])
        hits = self.vector_store.search(embedding[0], k=top_k)
        sources = self._build_sources(hits)

        if not sources:
            return ChatResponse(
                question=question,
                answer="I couldn't find relevant reviews in the dataset for that question.",
                sources=[],
            )

        if self._client:
            try:
                answer = self._answer_with_llm(question, sources)
            except Exception as e:
                logger.warning("LLM chat failed: %s. Using fallback.", e)
                answer = self._answer_with_fallback(question, sources)
        else:
            answer = self._answer_with_fallback(question, sources)

        return ChatResponse(question=question, answer=answer, sources=sources)

    def _build_sources(self, hits: list[dict]) -> list[ChatSource]:
        sources = []
        for hit in hits:
            review = self.review_map.get(hit["id"])
            analysis = self.analysis_map.get(hit["id"])
            sources.append(
                ChatSource(
                    review_id=hit["id"],
                    source=review.source.value if review else "unknown",
                    text=review.original_text if review else hit["text"],
                    score=hit["score"],
                    category=analysis.category.value if analysis else None,
                    sentiment=analysis.sentiment.value if analysis else None,
                    intent=analysis.intent.value if analysis else None,
                    themes=analysis.themes if analysis else [],
                )
            )
        return sources

    def _format_context(self, sources: list[ChatSource]) -> str:
        lines = []
        for src in sources:
            meta = []
            if src.category:
                meta.append(f"category={src.category}")
            if src.sentiment:
                meta.append(f"sentiment={src.sentiment}")
            if src.intent:
                meta.append(f"intent={src.intent}")
            if src.themes:
                meta.append(f"themes={', '.join(src.themes)}")
            meta_str = f" ({'; '.join(meta)})" if meta else ""
            lines.append(f"[{src.review_id}] [{src.source}]{meta_str}: \"{src.text}\"")
        return "\n".join(lines)

    def _answer_with_llm(self, question: str, sources: list[ChatSource]) -> str:
        context = self._format_context(sources)
        summary = ""
        if self.report:
            summary = (
                f"\nDataset summary ({self.report.total_reviews} reviews): "
                f"{'; '.join(self.report.key_insights_summary[:3])}"
            )

        prompt = f"""You are an analyst assistant for a music streaming platform's user feedback dataset.
Answer the user's question using ONLY the reviews and summary below.
Be concise, specific, and cite review IDs in brackets when referencing feedback.
If the dataset does not contain enough evidence, say what is known and what is missing.

{summary}

Relevant reviews:
{context}

User question: {question}"""

        response = self._client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    def _answer_with_fallback(self, question: str, sources: list[ChatSource]) -> str:
        sentiments = Counter(s.sentiment for s in sources if s.sentiment)
        categories = Counter(s.category for s in sources if s.category)
        themes: Counter = Counter()
        for src in sources:
            themes.update(src.themes)

        top_themes = [t for t, _ in themes.most_common(4)]
        top_category = categories.most_common(1)[0][0] if categories else "mixed feedback"
        dominant_sentiment = sentiments.most_common(1)[0][0] if sentiments else "mixed"

        lines = [
            f"Based on **{len(sources)} relevant reviews** from the dataset:",
            "",
            f"**Main theme:** {top_category} with **{dominant_sentiment}** sentiment.",
        ]
        if top_themes:
            lines.append(f"**Common topics:** {', '.join(top_themes)}.")
        lines.append("")
        lines.append("**What users are saying:**")
        for src in sources[:3]:
            lines.append(f'- [{src.review_id}] ({src.source}): "{src.text}"')
        if self.report and any(kw in question.lower() for kw in ("discovery", "recommend", "repeat", "ux", "mood")):
            lines.append("")
            lines.append("**From overall insights:**")
            for insight in self.report.key_insights_summary[:2]:
                lines.append(f"- {insight}")
        return "\n".join(lines)
