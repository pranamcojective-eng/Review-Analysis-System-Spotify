"""FastAPI server for the Review Discovery Engine."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.models import ChatResponse, PipelineResult
from src.pipeline import ReviewDiscoveryPipeline

app = FastAPI(
    title="Review Discovery Engine",
    description="AI-powered user feedback analysis for music streaming platforms",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline_result: PipelineResult | None = None
_pipeline: ReviewDiscoveryPipeline | None = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/")
def root():
    return {
        "service": "Review Discovery Engine",
        "endpoints": {
            "POST /pipeline/run": "Run full analysis pipeline",
            "GET /pipeline/status": "Get last pipeline run status",
            "GET /insights/report": "Get insight report",
            "GET /insights/clusters": "Get theme clusters",
            "GET /insights/sentiment": "Get sentiment distribution",
            "GET /insights/quotes": "Get sample quotes",
            "POST /search": "Semantic search similar reviews",
            "POST /chat": "Ask a question about the review dataset",
            "GET /health": "Health check",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_run": _pipeline_result is not None}


@app.post("/pipeline/run")
def run_pipeline_endpoint():
    global _pipeline_result, _pipeline
    try:
        _pipeline = ReviewDiscoveryPipeline()
        _pipeline_result = _pipeline.run()
        return {
            "status": "success",
            "raw_count": _pipeline_result.raw_count,
            "cleaned_count": _pipeline_result.cleaned_count,
            "duplicates_removed": _pipeline_result.duplicates_removed,
            "clusters": len(_pipeline_result.clusters),
            "log_entries": _pipeline_result.log_entries,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/status")
def pipeline_status():
    if not _pipeline_result:
        return {"status": "not_run", "message": "Pipeline has not been run yet. POST /pipeline/run first."}
    return {
        "status": "completed",
        "raw_count": _pipeline_result.raw_count,
        "cleaned_count": _pipeline_result.cleaned_count,
        "duplicates_removed": _pipeline_result.duplicates_removed,
        "log_entries": _pipeline_result.log_entries[-10:],
    }


@app.get("/insights/report")
def get_report():
    if not _pipeline_result:
        raise HTTPException(status_code=404, detail="Run pipeline first: POST /pipeline/run")
    return _pipeline_result.report.model_dump(mode="json")


@app.get("/insights/clusters")
def get_clusters():
    if not _pipeline_result:
        raise HTTPException(status_code=404, detail="Run pipeline first")
    return [c.model_dump() for c in _pipeline_result.clusters]


@app.get("/insights/sentiment")
def get_sentiment():
    if not _pipeline_result:
        raise HTTPException(status_code=404, detail="Run pipeline first")
    return _pipeline_result.report.sentiment_distribution


@app.get("/insights/quotes")
def get_quotes():
    if not _pipeline_result:
        raise HTTPException(status_code=404, detail="Run pipeline first")
    return _pipeline_result.report.sample_quotes


@app.post("/search")
def semantic_search(req: SearchRequest):
    if not _pipeline or not _pipeline.vector_store:
        raise HTTPException(status_code=404, detail="Run pipeline first to build vector index")
    embedding = _pipeline.embedding_service.encode([req.query])
    results = _pipeline.vector_store.search(embedding[0], k=req.k)
    return {"query": req.query, "results": results}


@app.post("/chat")
def chat_with_reviews(req: ChatRequest) -> ChatResponse:
    if not _pipeline or not _pipeline.chat_agent:
        raise HTTPException(status_code=404, detail="Run pipeline first to enable chat")
    return _pipeline.chat_agent.ask(req.question, top_k=req.top_k)
