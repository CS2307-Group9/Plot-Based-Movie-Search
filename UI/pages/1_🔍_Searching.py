import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\..')))

import json
from Indexing.BERT import *
from Indexing.BM25 import *
from Indexing.TF_IDF import *
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------
# Must be the first Streamlit command
# ----------------------------------------
st.set_page_config(page_title="Rovie - Movie Search", page_icon="🎬", layout="wide")

# ----------------------------------------
# Cached initialization (run once)
# ----------------------------------------
@st.cache_resource
def init_resources():

    indexing_configs = [
        ("BM25",
          None,
          "Indexing\\bm25_1\\bm25_index.pkl",
          1.0),
        ("TF_IDF",
          None,
          1,
          "Indexing\\tfidf_baseline_1\\tfidf_index"),  
        ("TF_IDF_sublinear",
          None,
          0,
          "Indexing\\tfidf_sublinearIDF_0\\tfidf_index"),  
        ("BERT",
          "BAAI/bge-large-en-v1.5",
          0.222,
          "Indexing\\bge-large-en-v1.5"),
        ("DistilBERT",
          "sentence-transformers/msmarco-distilbert-base-tas-b",
          0.333,
          "Indexing\\msmarco-distilbert-base-tas-b"),
        ("ROBERTA",
          "sentence-transformers/msmarco-roberta-base-v3",
          1.0,
          "Indexing\\msmarco-roberta-base-v3"),
    ]

    with open('Datasets\\imdb2plot.json', encoding="utf8") as f:
        imdb2plot = json.load(f)

    with open('Datasets\\imdb2title.json', encoding="utf8") as f:
        imdb2title = json.load(f)

    mad2plot_data = {}
    for key in imdb2title:
        if imdb2plot.get(key) is not None:
            mad2plot_data[key] = imdb2plot[key]

    indexing_stores = {}
    for method_id in range(0, len(indexing_configs)):
        config = indexing_configs[method_id]
        method_name = config[0]

        if "BM25" in method_name:
            
            BM25_collection = Collection()
            BM25_collection.load(imdb2plot)
            BM25_preprocessing = Preprocessing()
            #BM25_preprocessing.process(BM25_collection)
            BM25_indexing = Indexing(indexing_configs[method_id][2])
            #BM25_indexing.index(BM25_preprocessing)
            BM25_retrieval = Retrieval()

            indexing_stores[method_name] = (BM25_collection, BM25_preprocessing, BM25_indexing, BM25_retrieval)
        elif "TF_IDF" in method_name:
            indexing_stores[method_name] = TFIDFPipeline.load_local(config[3])
        else:
            indexing_stores[method_name] = BERT_Indexing(mad2plot_data,
                            chunker_model_name='all-mpnet-base-v2',
                            embed_model_name=config[1],
                            chunking_similarity_threshold=config[2],
                            index_path=config[3])

    df = pd.DataFrame({
        "imdb": imdb2title.keys(),
        "title": pd.Series(imdb2title),
        "plot": pd.Series(imdb2plot),
    })
    df.set_index('imdb', inplace=True, drop=True)
    
    return df, 25, indexing_configs, indexing_stores

df, K, indexing_configs, indexing_stores = init_resources()

# ----------------------------------------
# UI
# ----------------------------------------
st.title("🍿 Rovie — Movie Search by Plot")
st.markdown("Compare retrieval methods side-by-side. Enter a plot and pick methods.")

query = st.text_area(
    "Enter a plot description:",
    height=100,
    placeholder="e.g. a man wakes up in a virtual world controlled by machines..."
)

selected_methods = st.multiselect(
    "Select retrieval methods to compare:",
    [config[0] for config in indexing_configs],
    default=[indexing_configs[1][0], indexing_configs[2][0], indexing_configs[3][0]]
)

# ----------------------------------------
# Retrieval functions
# ----------------------------------------
def retrieve(method_name, query, K):

    index_store = indexing_stores[method_name]

    ranked_imdbs = []
    ranked_scores = []

    if "BM25" in method_name:
        BM25_collection, BM25_preprocessing, BM25_indexing, BM25_retrieval = index_store

        indexs, scores = BM25_retrieval.retrieve(BM25_indexing, BM25_preprocessing, query)
        
        ranked_imdbs = [BM25_collection.imdbs[id] for id in indexs]
        ranked_scores = scores
    else:
        retrieved_docs = index_store(query)[:K]
        for imdb, score, docs in retrieved_docs:
            ranked_imdbs.append(imdb)
            ranked_scores.append(score)

    results = df.loc[ranked_imdbs]
    results['score'] = ranked_scores
    
    return results

# Truncate titles and plots for consistent height
def truncate_text(text, max_len=35):
    return text if len(text) <= max_len else text[:max_len - 1] + "…"

# ----------------------------------------
# Display side-by-side with equal heights
# ----------------------------------------
if query.strip():
    st.markdown("### 🔎 Search Results Comparison")

    n = max(1, len(selected_methods))
    cols = st.columns([1] * n)

    for i, method in enumerate(selected_methods):
        with cols[i]:
            #short_name = display_names.get(method, method)
            st.markdown(f"**⚙️ {method}**", unsafe_allow_html=True)
            st.write("")  # Tiny spacer

            results = retrieve(method, query, K)

            for _, row in results.iterrows():
                short_title = truncate_text(row["title"], max_len=50)
                short_plot = truncate_text(row["plot"], max_len=1000000)
                exp_label = f"🎬 {short_title}"

                with st.expander(exp_label):
                    st.markdown(f"**Full title:** {row['title']}")
                    st.write(short_plot)  # Use truncated plot for consistency

else:
    st.info("Enter a plot above and select retrieval methods to start comparing results.")