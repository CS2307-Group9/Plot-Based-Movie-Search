import json
import bm25s
import Stemmer

class Collection:
    def __init__(self):
        self.mads = []
        self.plots = []
        
    def load_movie_plots(self, json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.mads.extend(data.keys())
            self.plots.extend(data.values())

    def get_mads_from_ids(self, ids: list):
        return [self.mads[id] for id in ids]

class Preprocessing:
    def __init__(self, collection: Collection):
        self.collection = collection

        self.stemmer = Stemmer.Stemmer("english")
        self.corpus_tokens = []

    def process(self):
        corpus = self.collection.plots
        self.corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=self.stemmer)

    def process_query(self, query: str):
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        return query_tokens

class Scoring:
    def __init__(self, preprocessing: Preprocessing):
        self.preprocessing = preprocessing

        self.K = 5
        self.retriever = bm25s.BM25()
        self.retriever.index(self.preprocessing.corpus_tokens)
        pass

    def calculate(self, query: str):
        query_tokens = preprocessing.process_query(query)
        results1, scores1 = self.retriever.retrieve(query_tokens, k=self.K)

        ranks = []
        scores = []
        for i in range(results1.shape[1]):
            ranks.append(int(results1[0, i]))
            scores.append(float(scores1[0, i]))

        return ranks, scores
    
class Evaluation:
    def __init__(self, collection: Collection, scoring: Scoring):
        self.collection = collection
        self.scoring = scoring

    def evaluate(self, json_path: str):
        eval_results = dict()
        with open(json_path, 'r') as f:
            data = json.load(f)

            for mad in data.keys():

                mad_results = []

                for query_id, query in enumerate(data[mad]):
                    ranks, _ = scoring.calculate(query)
                    mads = collection.get_mads_from_ids(ranks)

                    mad_results.append(mads)

                eval_results[mad] = mad_results

        return eval_results

if __name__ == "__main__":

    collection = Collection()
    collection.load_movie_plots("../../Datasets/MovieBench/data/mad2plot.json")
  
    preprocessing = Preprocessing(collection)
    preprocessing.process()

    scoring = Scoring(preprocessing)
    
    evaluation = Evaluation(collection, scoring)
    eval_results = evaluation.evaluate("../../Datasets/queries.json")
    
    with open("evaluate.json", "w") as f:
        json.dump(eval_results, f)
    
    # Test inference

    query = "Movie where teenagers at a magical boarding school compete in a dangerous international tournament, featuring dragons, underwater challenges, and a mysterious maze."
    print("query:", query)
    ranks, scores = scoring.calculate(query)

    for rank, score in zip(ranks, scores):
        print(collection.mads[rank], score)
    
    