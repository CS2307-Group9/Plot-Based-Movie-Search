from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from collections import defaultdict
import json
import os

# import nltk
# nltk.download('punkt_tab')

class SemanticChunker:
    def __init__(self, model, similarity_threshold=0.1):
        self.similarity_threshold = similarity_threshold
        self.model = model

    def __call__(self, text):
        sentences = sent_tokenize(text)
        embeddings = self.model.encode(sentences)

        # Group sentences based on cosine similarity threshold
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i]], [embeddings[i-1]])[0][0]
            if sim > self.similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                # print('chunk', " ".join(current_chunk))
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]

        chunks.append(" ".join(current_chunk))
        return chunks
    
class BERT_Indexing:
    def __init__(self, mad2plot_data, \
                 chunker_model_name='all-mpnet-base-v2', \
                 embed_model_name="sentence-transformers/msmarco-distilbert-base-tas-b", \
                 chunking_similarity_threshold=0.1, \
                 index_path=None):

        self.chunker_model = SentenceTransformer(chunker_model_name)
        model_kwargs = {'device': 'cpu', 'trust_remote_code': True,}
        self.embed_model = HuggingFaceEmbeddings(model_name=embed_model_name, model_kwargs=model_kwargs)
        # self.embed_model = HuggingFaceEmbeddings(model_name="retriever-chunk-model") # Use when have saved embedding
        self.semantic_chunker = SemanticChunker(self.chunker_model, chunking_similarity_threshold)
        self.build_index(index_path, mad2plot_data)

    def build_index(self, index_path, mad2plot_data):
        
        if index_path == None:
        
            self.data_chunks = []
            titles = []
            for key, value in tqdm(list(mad2plot_data.items())):
                chunks = self.semantic_chunker(value)
                titles.append(key)
                self.data_chunks.append(chunks)

            texts = []
            metadatas = []
            for movie_plots, title in zip(self.data_chunks, titles):
                for plot in movie_plots:
                    texts.append(plot)
                    metadatas.append({"title": title})

            self.semantic_chunk_vectorstore = FAISS.from_texts(texts, embedding=self.embed_model, metadatas=metadatas, distance_strategy=DistanceStrategy.COSINE)
            self.len_chunks = len(texts)
        else:
            self.semantic_chunk_vectorstore = FAISS.load_local(index_path, self.embed_model,allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.COSINE)
            self.len_chunks = len(self.semantic_chunk_vectorstore.index_to_docstore_id)

    def __call__(self, query):

        expanded_queries = [query]
        all_chunks = []

        for expanded_query in expanded_queries:
            chunks = self.semantic_chunk_vectorstore.similarity_search_with_score(expanded_query, k=self.len_chunks)
            # Assuming this returns a list of (chunk, score) tuples
            all_chunks.extend(chunks)

        # Group chunks by document ID
        doc_scores = defaultdict(list)
        doc_chunks = defaultdict(list)

        for chunk, score in all_chunks:
            doc_id = chunk.metadata['title']  # Adjust field name if needed
            doc_scores[doc_id].append(score)
            doc_chunks[doc_id].append((chunk, score))

        # Select max score per document
        ranked_docs = []
        for doc_id, scores in doc_scores.items():

            max_score_index = scores.index(max(scores))
            max_score = max(scores)
            ranked_docs.append((doc_id, max_score, doc_chunks[doc_id][max_score_index]))

        # Sort documents by max score (descending)
        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        return ranked_docs
    
if False:#__name__ == "__main__":
    
    with open('Datasets\\imdb2plot.json', encoding="utf8") as f:
        final_id2title = json.load(f)

    with open('Datasets\\imdb2title.json', encoding="utf8") as f:
        final_data = json.load(f)

    mad2plot_data = {}
    for key in final_id2title:
        if final_data.get(key)  is not None:
            mad2plot_data[final_id2title[key]] = final_data[key]

    embed_model_name = "BAAI/bge-large-en-v1.5" # sentence-transformers/msmarco-distilbert-base-tas-b, BAAI/bge-large-en-v1.5, nomic-ai/nomic-embed-text-v1
    threshold = 0.222

    indexing = BERT_Indexing(mad2plot_data,
                        chunker_model_name='all-mpnet-base-v2',
                        embed_model_name=embed_model_name,
                        chunking_similarity_threshold=threshold,
                        index_path='Indexing\\BAAI\\bge-large-en-v1.5')
    
    movie_query = "Movie where teenagers at a magical boarding school compete in a dangerous international tournament, featuring dragons, underwater challenges, and a mysterious maze."

    retrieved_docs = indexing(movie_query)[:25]
    retrieved_titles = []
    for title, score, docs in retrieved_docs:
        retrieved_titles.append(title)
    # if movie_title not in results:
    #     results[movie_title] = []
    # results[movie_title].append(retrieved_titles)
    print(retrieved_titles)