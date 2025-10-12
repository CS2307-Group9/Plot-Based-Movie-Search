import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import plotly.express as px
import random

title = "Evaluation"
st.set_page_config(page_title=title, page_icon="📊")

st.markdown("# " + title)
st.sidebar.header(title)
st.write(
    """Describe a movie. We'll find it."""
)

INDEXING_METHODS = ["TF-IDF", "LSI", "BERT", "BM25"]

MOVIES = [
            "Movie A", "Movie B", "Movie C", "Movie D", "Movie E",
            "Movie F", "Movie G", "Movie H", "Movie I", "Movie J"
        ]

try:

    query = st.text_area(
        "Query",
        "It was the best of times, it was the worst of times, it was the age of "
        "wisdom, it was the age of foolishness, it was the epoch of belief, it "
        "was the epoch of incredulity, it was the season of Light, it was the "
        "season of Darkness, it was the spring of hope, it was the winter of "
        "despair, (...)",
    )

    indexing_methods = st.multiselect(
        "Select indexing methods",
        INDEXING_METHODS,
        default=["TF-IDF", "LSI", "BM25"],
    )
    if not indexing_methods:
        st.error("Please select at least one method.")

    similarity_method = st.selectbox(
        "Select a similarity methods",
        ("Cosine"),
    )
    if not similarity_method:
        st.error("Please select a method.")

    _, middle_search, _ = st.columns(3)
    if middle_search.button("Search", icon="🔎", use_container_width=True):
        
        retrieval_results = {}
        for indexing_method in indexing_methods:
            retrieval_results[indexing_method] = random.sample(MOVIES, 5)

        # retrieval_results = {
        #     "TF-IDF": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"],
        #     "LSI":    ["Movie C", "Movie G", "Movie H", "Movie A", "Movie F"],
        #     "BERT":   ["Movie J", "Movie B", "Movie H", "Movie C", "Movie D"]
        # }

        records = []
        for movie in MOVIES:
            for method, top_movies in retrieval_results.items():
                if movie in top_movies:
                    rank = top_movies.index(movie) + 1
                else:
                    rank = None  # You can use None to hide, or 6 to show "Not Ranked"
                records.append({"Movie": movie,
                                "Method": method,
                                "Rank": rank,
                                "Label": movie
                                })

        df = pd.DataFrame(records)

        fig = px.line(df, x="Method", y="Rank", color="Movie", markers=True,
                    text="Label",
                    title="Movie Ranking Across Retrieval Methods",
                    labels={"Rank": "Rank (lower is better)"},
                    hover_data={"Movie": True, "Rank": True})

        fig.update_layout(yaxis=dict(autorange="reversed", dtick=1))

        fig.update_traces(textposition="top center", mode="lines+markers+text")

        st.plotly_chart(fig, use_container_width=True)


        
except URLError as e:
    st.error(e.reason)
