import json

from metrics import *

class Metric(Enum):
    P = 1
    R = 2
    AP = 3
    RP = 4
    RR = 5
    NDCG = 6

class MeanEvaluation:
    @staticmethod
    def compute(relevances_set: list[list[bool]], K: int, R: int, metric: Metric):
        score_sum = 0

        for relevances in relevances_set:
            if metric == Metric.P:
                score_sum += P_K.compute(relevances, K)
            elif metric == Metric.R:
                score_sum += R_K.compute(relevances, K, R)
            elif metric == Metric.AP:
                score_sum += AP_K.compute(relevances, K)
            elif metric == Metric.RP:
                score_sum += RP_K.compute(relevances, R)
            elif metric == Metric.RR:
                score_sum += RR_K.compute(relevances, K)
            elif metric == Metric.NDCG:
                score_sum += NDCG_K.compute(relevances, K, R)

        return score_sum / len(relevances_set)
    
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

    with open(output_json_path, 'r') as f:
        ranked_result = json.load(f)

        relevances_set = Ranked2Relevance.convert(ranked_result)
        
        m_p = MeanEvaluation.compute(relevances_set, K, R, Metric.P)
        print(f"Mean-P@{K}:", m_p)

        m_r = MeanEvaluation.compute(relevances_set, K, R, Metric.R)
        print(f"Mean-R@{K}:", m_r)

        m_ap = MeanEvaluation.compute(relevances_set, K, R, Metric.AP)
        print(f"Mean-AP@{K}:", m_ap)
        
        m_rp = MeanEvaluation.compute(relevances_set, K, R, Metric.RP)
        print(f"Mean-RP@{K}", m_rp)

        m_rr = MeanEvaluation.compute(relevances_set, K, R, Metric.RR)
        print(f"Mean-RR@{K}", m_rr)

        m_ndcg = MeanEvaluation.compute(relevances_set, K, R, Metric.NDCG)
        print(f"Mean-NDCG@{K}", m_ndcg)

    import numpy as np
    keys = [
        "0.0",
        "0.1111111111111111",
        "0.2222222222222222",
        "0.3333333333333333",
        "0.4444444444444444",
        "0.5555555555555556",
        "0.6666666666666666",
        "0.7777777777777777",
        "0.8888888888888888",
        "1.0"
    ]

    output = {
        "P": {str(k): m_p for k in keys},
        "R": {str(k): m_r for k in keys},
        "AP": {str(k): m_ap for k in keys},
        "RP": {str(k): m_rp for k in keys},
        "RR": {str(k): m_rr for k in keys},
        "NDCG": {str(k): m_ndcg for k in keys},
    }
    
    with open("Output\\eval5000_25_BM25.json", "w") as f:
        json.dump(output, f, indent=4)

    pass