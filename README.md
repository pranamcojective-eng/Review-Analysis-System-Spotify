# Review Discovery Engine

AI-powered user feedback analysis pipeline for music streaming platforms (Spotify-like). Ingests reviews from multiple sources, processes text, classifies with LLM, clusters themes with embeddings + FAISS, and generates actionable insights.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Ingestion      │───▶│  Processing      │───▶│  LLM Analysis   │───▶│  Insight Gen     │
│  App/Play Store │    │  Clean + Dedup   │    │  Classify       │    │  Top struggles   │
│  Reddit/Twitter │    │  Remove noise    │    │  Sentiment      │    │  Segments        │
└─────────────────┘    └──────────────────┘    │  Intent/Themes  │    │  Summary         │
                                                  └────────┬────────┘    └──────────────────┘
                                                           │
                                                  ┌────────▼────────┐
                                                  │ Embeddings+FAISS│
                                                  │ Theme Clustering│
                                                  └─────────────────┘
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set OpenAI API key for real LLM analysis
copy .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 4. Run pipeline via CLI
python run_pipeline.py

# 5. Launch dashboard
streamlit run dashboard/app.py

# 6. Or start API server
uvicorn src.api.main:app --reload --port 8000
```

## Outputs

After running the pipeline, check `./output/`:

| File | Description |
|------|-------------|
| `insight_report.json` | Full structured insight report |
| `review_analyses.json` | Per-review classification results |
| `vector_store_meta.json` | FAISS index metadata |
| `pipeline_run.json` | Execution summary |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/pipeline/run` | Run full pipeline |
| GET | `/insights/report` | Full insight report |
| GET | `/insights/clusters` | Theme clusters |
| GET | `/insights/sentiment` | Sentiment distribution |
| GET | `/insights/quotes` | Sample user quotes |
| POST | `/chat` | Ask questions about the dataset (`{"question": "...", "top_k": 5}`) |

## Chat Q&A

The **Ask About Reviews** tab in the dashboard lets users type questions and get answers grounded in the review dataset. The chat agent:

1. Retrieves the most relevant reviews from the dataset
2. Enriches them with category, sentiment, and intent metadata
3. Synthesizes an answer (GPT-4o-mini with API key, or rule-based fallback)

Example API call:

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What UX issues do users report?\"}"
```

## LLM Mode

- **With `OPENAI_API_KEY`**: Uses GPT-4o-mini for classification, sentiment, intent, and theme extraction.
- **Without API key**: Uses a rule-based fallback analyzer (works out of the box with mock data).

## Data Sources

Currently uses mock data in `data/mock_reviews.json` simulating:
- App Store reviews
- Play Store reviews
- Reddit discussions
- Twitter/X posts

Includes intentional duplicates to demonstrate deduplication.

## Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app** and select the repository:
   `pranamcojective-eng/Review-Analysis-System-Spotify`
3. Use these deployment settings:

| Setting | Value |
|---------|-------|
| **Branch** | `main` |
| **Main file path** | `streamlit_app.py` |
| **Python version** | `3.11` |

4. (Optional) Open **Advanced settings → Secrets** and paste:

```toml
OPENAI_API_KEY = "sk-your-key-here"
OPENAI_MODEL = "gpt-4o-mini"
```

   Without `OPENAI_API_KEY`, the app uses the built-in rule-based analyzer.

5. Click **Deploy**. The first pipeline run downloads the embedding model (~80 MB) and may take 1–2 minutes.

### Local Streamlit run

```bash
streamlit run streamlit_app.py
# or
streamlit run dashboard/app.py
```
