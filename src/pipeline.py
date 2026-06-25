import json
import logging
from datetime import datetime
from pathlib import Path

from src.analysis.chat_agent import ReviewChatAgent
from src.analysis.clusterer import ThemeClusterer
from src.analysis.embeddings import EmbeddingService
from src.analysis.llm_analyzer import LLMAnalyzer
from src.analysis.vector_store import FAISSVectorStore
from src.config import OUTPUT_DIR, setup_logging
from src.ingestion.sources import create_default_ingestion_manager
from src.insights.generator import InsightGenerator
from src.models import PipelineResult
from src.processing.cleaner import TextCleaner
from src.processing.deduplicator import Deduplicator

logger = setup_logging()


class ReviewDiscoveryPipeline:
    """End-to-end pipeline orchestrating all layers."""

    def __init__(self):
        self.ingestion = create_default_ingestion_manager()
        self.cleaner = TextCleaner(remove_emojis=True)
        self.deduplicator = Deduplicator(similarity_threshold=0.85)
        self.analyzer = LLMAnalyzer()
        self.embedding_service = EmbeddingService()
        self.clusterer = ThemeClusterer(self.embedding_service)
        self.insight_generator = InsightGenerator()
        self.vector_store: FAISSVectorStore | None = None
        self.reviews: list = []
        self.analyses: list = []
        self.chat_agent: ReviewChatAgent | None = None
        self.log_entries: list[str] = []

    def _log(self, message: str) -> None:
        self.log_entries.append(f"[{datetime.now().isoformat()}] {message}")
        logger.info(message)

    def run(self) -> PipelineResult:
        self.log_entries = []
        self._log("=== Review Discovery Pipeline Started ===")

        # Layer 1: Ingestion
        self._log("Layer 1: Data Ingestion")
        raw_reviews = self.ingestion.ingest_all()
        self._log(f"  Ingested {len(raw_reviews)} raw reviews")

        # Layer 2: Processing
        self._log("Layer 2: Text Processing")
        cleaned = self.cleaner.process_batch(raw_reviews)
        unique, dupes_removed = self.deduplicator.deduplicate(cleaned)
        self._log(f"  Cleaned: {len(cleaned)}, Unique: {len(unique)}, Duplicates removed: {dupes_removed}")

        # Layer 3: LLM Analysis
        self._log("Layer 3: LLM Analysis")
        analyses = self.analyzer.analyze_batch(unique)
        self._log(f"  Analyzed {len(analyses)} reviews")

        # Layer 3b: Embeddings + Clustering + Vector Store
        self._log("Layer 3b: Embeddings, Clustering & Vector Store")
        clusters, embeddings = self.clusterer.cluster(unique, analyses)
        self._log(f"  Created {len(clusters)} theme clusters")

        if len(embeddings) > 0:
            self.vector_store = FAISSVectorStore(dimension=embeddings.shape[1])
            self.vector_store.add(
                ids=[r.id for r in unique],
                embeddings=embeddings,
                texts=[r.cleaned_text for r in unique],
            )
            self._log(f"  FAISS index built with {self.vector_store.size} vectors")

        # Layer 4: Insight Generation
        self._log("Layer 4: Insight Generation")
        report = self.insight_generator.generate(unique, analyses, clusters)
        self._log("  Insight report generated")

        self.reviews = unique
        self.analyses = analyses
        if self.vector_store:
            self.chat_agent = ReviewChatAgent(
                self.embedding_service,
                self.vector_store,
                unique,
                analyses,
                report,
            )
            self._log("  Review chat agent ready")

        result = PipelineResult(
            raw_count=len(raw_reviews),
            cleaned_count=len(unique),
            duplicates_removed=dupes_removed,
            analyses=analyses,
            clusters=clusters,
            report=report,
            log_entries=self.log_entries,
        )

        self._save_outputs(result, unique)
        self._log("=== Pipeline Complete ===")
        return result

    def _save_outputs(self, result: PipelineResult, reviews) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        report_path = OUTPUT_DIR / "insight_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result.report.model_dump(mode="json"), f, indent=2, default=str)
        self._log(f"  Saved report: {report_path}")

        analyses_path = OUTPUT_DIR / "review_analyses.json"
        with open(analyses_path, "w", encoding="utf-8") as f:
            json.dump([a.model_dump() for a in result.analyses], f, indent=2)
        self._log(f"  Saved analyses: {analyses_path}")

        if self.vector_store:
            meta_path = OUTPUT_DIR / "vector_store_meta.json"
            self.vector_store.save_metadata(str(meta_path))

        logs_path = OUTPUT_DIR / "pipeline_run.json"
        with open(logs_path, "w", encoding="utf-8") as f:
            json.dump({
                "raw_count": result.raw_count,
                "cleaned_count": result.cleaned_count,
                "duplicates_removed": result.duplicates_removed,
                "log_entries": result.log_entries,
            }, f, indent=2)
        self._log(f"  Saved run log: {logs_path}")


def run_pipeline() -> PipelineResult:
    pipeline = ReviewDiscoveryPipeline()
    return pipeline.run()
