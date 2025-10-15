import json
import bm25s
import Stemmer
import os
import pickle

class Collection:
    def __init__(self):
        self.imdbs = []
        self.plots = []
        
    def load(self, imdb2plot: dict):
        self.imdbs.extend(imdb2plot.keys())
        self.plots.extend(imdb2plot.values())

    def get_imdbs_from_ids(self, ids: list):
        return [self.imdbs[id] for id in ids]

class Preprocessing:
    def __init__(self):
        self.stemmer = Stemmer.Stemmer("english")
        self.corpus_tokens = []

    def process(self, collection: Collection):
        self.corpus_tokens = bm25s.tokenize(collection.plots, stopwords="en", stemmer=self.stemmer)

    def process_query(self, query: str):
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        return query_tokens

class Indexing:
    def __init__(self, path: str = None):
        if path is None:
            self.retriever = bm25s.BM25()
        else:
            with open(path, "rb") as f:
                self.retriever = pickle.load(f)
        pass

    def index(self, preprocessing: Preprocessing):
        self.retriever.index(preprocessing.corpus_tokens)

class Retrieval:
    def __init__(self):
        self.K = 25
        
    def retrieve(self, indexing: Indexing, preprocessing: Preprocessing, query: str):
        query_tokens = preprocessing.process_query(query)
        ranks1, scores1 = indexing.retriever.retrieve(query_tokens, k=self.K)

        ranks = []
        scores = []
        for i in range(ranks1.shape[1]):
            ranks.append(int(ranks1[0, i]))
            scores.append(float(scores1[0, i]))

        return ranks, scores
    
class Evaluation:
    def __init__(self):
        pass

    def evaluate(self, indexing: Indexing, preprocessing: Preprocessing, retrieval: Retrieval, json_path: str):
        eval_results = dict()
        with open(json_path, 'r') as f:
            data = json.load(f)

            for imdb in data.keys():

                imdbs_results = []

                for query in data[imdb]:
                    ranks, _ = retrieval.retrieve(indexing, preprocessing, query)
                    imdbs_result = collection.get_imdbs_from_ids(ranks)

                    imdbs_results.append(imdbs_result)

                eval_results[imdb] = imdbs_results

        return eval_results

if __name__ == "__main__":

    with open('Datasets\\imdb2plot.json', encoding="utf8") as f:
        imdb2plot = json.load(f)

    with open('Datasets\\imdb2title.json', encoding="utf8") as f:
        imdb2title = json.load(f)
    
    BM25_collection = Collection()
    BM25_collection.load(imdb2plot)
    BM25_preprocessing = Preprocessing()
    BM25_preprocessing.process(BM25_collection)
    BM25_indexing = Indexing("Indexing\\bm25_1\\bm25_index.pkl")
    BM25_indexing.index(BM25_preprocessing)

    # # --- Save index ---
    # with open("Indexing\\bm25_index.pkl", "wb") as f:
    #     pickle.dump(BM25_indexing.retriever, f)

    # --- Load index later ---
    # with open("bm25_index.pkl", "rb") as f:
    #     bm25_loaded = pickle.load(f)