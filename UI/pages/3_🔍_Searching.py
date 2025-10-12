import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import plotly.express as px
import random
import json
import pandas as pd
import streamlit as st
import sys
import os
import importlib.util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\..')))
from Indexing.BM25 import *
from Indexing.BERT import *

with open('Datasets\\imdb2plot.json', encoding="utf8") as f:
    imdb2plot = json.load(f)

with open('Datasets\\imdb2title.json', encoding="utf8") as f:
    imdb2title = json.load(f)

title = "Searching"
st.set_page_config(page_title=title, page_icon="🔍")

st.markdown("# " + title)
st.sidebar.header(title)
st.write(
    """Describe a movie. We'll find it."""
)

MOVIES = list(imdb2title.keys())

try:

    query = st.text_area(
        "Query",
        "It was the best of times, it was the worst of times, it was the age of "
        "wisdom, it was the age of foolishness, it was the epoch of belief, it "
        "was the epoch of incredulity, it was the season of Light, it was the "
        "season of Darkness, it was the spring of hope, it was the winter of "
        "despair, (...)",
    )

    indexing_methods = [0, 1, 2, 3]

    labels = {
        1: "BERT",
        2: "DistilBERT",
        3: "ROBERTA",
        0: "BM25",
    }

    indexing_method = st.selectbox(
        "Select an indexing method:",
        options=list(labels.keys()),
        format_func=lambda x: labels[x]
    )

    embed_model_names = [
        None,
        "BAAI/bge-large-en-v1.5",
        "sentence-transformers/msmarco-distilbert-base-tas-b",
        "sentence-transformers/msmarco-roberta-base-v3"
    ]

    thresholds = [
        0,
        0.222,
        0.333,
        1.0
    ]

    index_paths = [
        None,
        "Indexing\\BAAI\\bge-large-en-v1.5",
        "Indexing\\msmarco-distilbert-base-tas-b",
        "Indexing\\msmarco-roberta-base-v3"
    ]

    if indexing_method != 0:
        mad2plot_data = {}
        for key in imdb2title:
            if imdb2plot.get(key) is not None:
                mad2plot_data[imdb2title[key]] = imdb2plot[key]

        #embed_model_name = "BAAI/bge-large-en-v1.5" # sentence-transformers/msmarco-distilbert-base-tas-b, BAAI/bge-large-en-v1.5, nomic-ai/nomic-embed-text-v1
        #threshold = 0.222

        indexing = BERT_Indexing(mad2plot_data,
                            chunker_model_name='all-mpnet-base-v2',
                            embed_model_name=embed_model_names[indexing_method],
                            chunking_similarity_threshold=thresholds[indexing_method],
                            index_path=index_paths[indexing_method])
    elif indexing_method == 0:
        collection = Collection()
        collection.load(imdb2plot)

        preprocessing = Preprocessing()
        preprocessing.process(collection)

        indexing = Indexing()
        indexing.index(preprocessing)

        retrieval = Retrieval()

    K = 25

    _, middle_search, _ = st.columns(3)
    if middle_search.button("Search", icon="🔎", use_container_width=True):

        ranked_imdbs = []
        ranked_scores = []

        if indexing_method != 0:
            retrieved_docs = indexing(query)[:K]
            for imdb, score, docs in retrieved_docs:
                ranked_imdbs.append(imdb)
                ranked_scores.append(score)
        elif indexing_method == 0:
            ranks, scores = retrieval.retrieve(indexing, preprocessing, query)
            ranked_imdbs = [collection.imdbs[rank] for rank in ranks]
            ranked_scores = scores
        
        df = pd.DataFrame(
        {
            "Rank": list(range(1, K + 1)),
            "Movie": [imdb2title[imdb] for imdb in ranked_imdbs],
            "Plot": [imdb2plot[imdb][:300] for imdb in ranked_imdbs],
            #"Score": ranked_scores
        })
        df.set_index(df.columns[0])
        st.table(df)

        
except URLError as e:
    st.error(e.reason)
