import json
from ranx import Qrels, Run, evaluate
import numpy as np

class Ranked2Relevance:
    @staticmethod
    def convert(ranked_result: dict):

        relevances_set = []

        for true_doc in ranked_result.keys():
            pred_docs_set = ranked_result[true_doc]

            for pred_docs in pred_docs_set:
                relevances = [True if pred_doc == true_doc else False for pred_doc in pred_docs]
                relevances_set.append(relevances)
                
        return relevances_set

if __name__ == "__main__":

    output_json_path = "Output/BM25.json"
    K = 25
    R = 1

    qrels_dict = {}
    run_dict = {}

    with open(output_json_path, 'r') as f:
        ranked_result = json.load(f)

        for true_doc in ranked_result.keys():
            pred_docs_set = ranked_result[true_doc]

            query_id = 1
            
            for pred_docs in pred_docs_set:

                true_relevances = { true_doc: 1 }

                pred_relevances = {}
                pseudo_relevances = np.linspace(1, 0, len(pred_docs), endpoint=False)
                
                for pred_doc, pseudo_relevance in zip(pred_docs, pseudo_relevances):
                    pred_relevances[pred_doc] = pseudo_relevance
                
                query_name = true_doc + "_" + str(query_id)
                qrels_dict[query_name] = true_relevances
                run_dict[query_name] = pred_relevances

                query_id += 1
                #break

    # qrels_dict = { "q_1": { "d_12": 5, "d_25": 3 },
    #             "q_2": { "d_11": 6, "d_22": 1 } }
    qrels = Qrels(qrels_dict)

    # run_dict = { "q_1": { "d_12": 0.9, "d_23": 0.8, "d_25": 0.7,
    #                     "d_36": 0.6, "d_32": 0.5, "d_35": 0.4  },
    #             "q_2": { "d_12": 0.9, "d_11": 0.8, "d_25": 0.7,
    #                     "d_36": 0.6, "d_22": 0.5, "d_35": 0.4  } }
    run = Run(run_dict)

    print(evaluate(qrels, run, ["precision@25", "recall@25", "map@25", "r-precision", "mrr@25", "ndcg@25"]))
