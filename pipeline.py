import json
import bm25s
import Stemmer
import os

class Collection:
    def __init__(self):
        self.imdbs = []
        self.plots = []
        
    def load(self, json_path: str):
        with open(json_path, 'r', encoding="utf8") as f:
            data = json.load(f)
            self.imdbs.extend(data.keys())
            self.plots.extend(data.values())

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
    def __init__(self):
        self.retriever = bm25s.BM25()
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

    dataset_dir = "Datasets"
    evaluation_dir = "Evaluation"
    output_dir = "Output"

    collection = Collection()
    collection.load(os.path.join(dataset_dir, "imdb2plot.json"))

    preprocessing = Preprocessing()
    preprocessing.process(collection)

    indexing = Indexing()
    indexing.index(preprocessing)

    retrieval = Retrieval()

    evaluation = Evaluation()
    eval_results = evaluation.evaluate(indexing, preprocessing, retrieval, os.path.join(os.path.join(dataset_dir, "new_queries.json")))
    
    with open(os.path.join(output_dir, "BM25.json"), "w") as f:
        json.dump(eval_results, f)
    
    # Test inference

    query = "Movie where teenagers at a magical boarding school compete in a dangerous international tournament, featuring dragons, underwater challenges, and a mysterious maze."
    print("query:", query)
    ranks, scores = retrieval.retrieve(indexing, preprocessing, query)

    for rank, score in zip(ranks, scores):
        print(collection.imdbs[rank], score)
    
    