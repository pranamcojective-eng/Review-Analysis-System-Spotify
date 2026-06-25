"""Streamlit dashboard for the Review Discovery Engine."""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ReviewDiscoveryPipeline  # noqa: E402

st.set_page_config(
    page_title="Review Discovery Engine",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Review Discovery Engine")
st.caption("AI-powered user feedback analysis for music streaming platforms")

if "result" not in st.session_state:
    st.session_state.result = None
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def run_pipeline():
    with st.spinner(
        "Running pipeline... (ingestion → processing → LLM analysis → clustering → insights). "
        "First run may take 1–2 minutes while the embedding model downloads."
    ):
        pipeline = ReviewDiscoveryPipeline()
        result = pipeline.run()
        st.session_state.pipeline = pipeline
        st.session_state.result = result
        st.session_state.chat_history = []
    st.success("Pipeline complete!")


col_run, col_info = st.columns([1, 3])
with col_run:
    if st.button("▶ Run Pipeline", type="primary", use_container_width=True):
        run_pipeline()

with col_info:
    if st.session_state.result:
        r = st.session_state.result
        st.info(
            f"Last run: {r.raw_count} raw → {r.cleaned_count} unique "
            f"({r.duplicates_removed} duplicates removed) · {len(r.clusters)} clusters"
        )
    else:
        st.info("Click **Run Pipeline** to analyze mock review data from App Store, Play Store, Reddit, and Twitter/X.")

if not st.session_state.result:
    st.markdown("""
    ### Architecture
    | Layer | Description |
    |-------|-------------|
    | **Ingestion** | Load reviews from App Store, Play Store, Reddit, Twitter (mock data) |
    | **Processing** | Clean text, remove emojis/noise, deduplicate |
    | **LLM Analysis** | Classify, extract sentiment, intent, themes (GPT-4 or rule-based fallback) |
    | **Embeddings + FAISS** | Vectorize reviews, cluster themes, enable semantic search |
    | **Insights** | Generate structured report with top struggles, segments, quotes |
    """)
    st.stop()

result = st.session_state.result
report = result.report

tab1, tab_chat, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "💭 Ask About Reviews",
    "🏷️ Theme Clusters",
    "💬 Quotes",
    "🔍 Semantic Search",
    "📋 Pipeline Logs",
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", report.total_reviews)
    c2.metric("Negative", report.sentiment_distribution.get("negative", 0))
    c3.metric("Positive", report.sentiment_distribution.get("positive", 0))
    c4.metric("Clusters", len(report.theme_clusters))

    st.subheader("Key Insights")
    for insight in report.key_insights_summary:
        st.markdown(f"- {insight}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Sentiment Distribution")
        if report.sentiment_distribution:
            fig = px.pie(
                names=list(report.sentiment_distribution.keys()),
                values=list(report.sentiment_distribution.values()),
                color_discrete_map={"positive": "#1DB954", "negative": "#E91429", "neutral": "#B3B3B3"},
            )
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Category Distribution")
        if report.category_distribution:
            fig = px.bar(
                x=list(report.category_distribution.values()),
                y=list(report.category_distribution.keys()),
                orientation="h",
                color=list(report.category_distribution.values()),
                color_continuous_scale="Greens",
            )
            fig.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Discovery Struggles")
    for item in report.top_discovery_struggles:
        st.markdown(f"- {item}")

    st.subheader("Recommendation Frustrations")
    for item in report.recommendation_frustrations:
        st.markdown(f"- {item}")

    st.subheader("Repetition Patterns")
    for item in report.repetition_patterns:
        st.markdown(f"- {item}")

    st.subheader("Emerging Unmet Needs")
    for item in report.emerging_unmet_needs:
        st.markdown(f"- {item}")

    st.subheader("User Segments")
    if report.user_segments:
        seg_df = pd.DataFrame(
            list(report.user_segments.items()), columns=["Segment", "Count"]
        )
        fig = px.bar(seg_df, x="Segment", y="Count", color="Count", color_continuous_scale="Blues")
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig, use_container_width=True)

with tab_chat:
    st.subheader("Ask About the Review Dataset")
    st.caption(
        "Type a question and get answers grounded in user reviews. "
        "The chat retrieves relevant feedback via semantic search, then synthesizes a response."
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Source reviews used"):
                    for src in message["sources"]:
                        st.markdown(
                            f"**[{src['review_id']}]** {src['source']} · "
                            f"score {src['score']:.2f} · {src.get('category', 'N/A')}"
                        )
                        st.markdown(f"> {src['text']}")

    if prompt := st.chat_input("e.g. Why do users struggle with music discovery?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.pipeline and st.session_state.pipeline.chat_agent:
                with st.spinner("Searching reviews and generating answer..."):
                    response = st.session_state.pipeline.chat_agent.ask(prompt)
                st.markdown(response.answer)
                if response.sources:
                    with st.expander("Source reviews used"):
                        for src in response.sources:
                            st.markdown(
                                f"**[{src.review_id}]** {src.source} · "
                                f"score {src.score:.2f} · {src.category or 'N/A'}"
                            )
                            st.markdown(f"> {src.text}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": [s.model_dump() for s in response.sources],
                })
            else:
                st.warning("Chat is unavailable. Run the pipeline first.")

    if st.session_state.chat_history and st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

with tab2:
    st.subheader("Theme Clusters")
    for cluster in report.theme_clusters:
        with st.expander(f"**{cluster.label}** — {cluster.review_count} reviews", expanded=False):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Category:** {cluster.dominant_category}")
            c2.markdown(f"**Sentiment:** {cluster.dominant_sentiment}")
            st.markdown(f"**Keywords:** {', '.join(cluster.theme_keywords)}")
            st.markdown("**Sample quotes:**")
            for q in cluster.sample_quotes:
                st.markdown(f"> {q}")

with tab3:
    st.subheader("Sample User Quotes")
    for quote in report.sample_quotes:
        sentiment_color = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
            quote["sentiment"], "⚪"
        )
        st.markdown(
            f"{sentiment_color} **{quote['source']}** · {quote['category']} · _{quote['intent']}_"
        )
        st.markdown(f"> {quote['text']}")
        st.divider()

with tab4:
    st.subheader("Semantic Search (FAISS)")
    query = st.text_input("Search similar feedback", placeholder="e.g. algorithm plays same songs")
    if query and st.session_state.pipeline and st.session_state.pipeline.vector_store:
        embedding = st.session_state.pipeline.embedding_service.encode([query])
        results = st.session_state.pipeline.vector_store.search(embedding[0], k=5)
        for r in results:
            st.markdown(f"**Score: {r['score']:.3f}** — `{r['id']}`")
            st.markdown(f"> {r['text']}")
            st.divider()

with tab5:
    st.subheader("Pipeline Execution Log")
    for entry in result.log_entries:
        st.text(entry)

    st.subheader("Intermediate Outputs")
    output_dir = PROJECT_ROOT / "output"
    if output_dir.exists():
        for f in sorted(output_dir.glob("*.json")):
            with st.expander(f.name):
                st.json(json.loads(f.read_text(encoding="utf-8")))
