from enum import Enum
import math

class P_K:
    @staticmethod
    def compute(relevances: list[bool], K: int):
        relevance_sum = 0

        for rank, relevance in enumerate(relevances, start=1):
            
            if rank > K:
                break

            if relevance:
                relevance_sum += 1

        return relevance_sum / K

class R_K:
    @staticmethod
    def compute(relevances: list[bool], K: int, R: int):
        relevance_sum = 0

        for rank, relevance in enumerate(relevances, start=1):
            
            if rank > K:
                break

            if relevance:
                relevance_sum += 1

        return relevance_sum / R

class AP_K:
    @staticmethod
    def compute(relevances: list[bool], K: int):
        precision_sum = 0

        relevance_count = 0
        for rank, relevance in enumerate(relevances, start=1):
            
            if rank > K:
                break

            if relevance:
                relevance_count += 1
                precision_sum += relevance_count / rank

        if relevance_count == 0:
            return 0
        
        return precision_sum / relevance_count
    
class RP_K:
    @staticmethod
    def compute(relevances: list[bool], R: int):
        relevance_sum = 0

        for rank, relevance in enumerate(relevances, start=1):
            
            if rank > R:
                break

            if relevance:
                relevance_sum += 1

        return relevance_sum / R
    
class RR_K:
    @staticmethod
    def compute(relevances: list[bool], K: int):
        rr = 0

        for rank, relevance in enumerate(relevances, start=1):
            
            if rank > K:
                break

            if relevance:
                rr = 1 / rank
                break
            
        return rr

class NDCG_K:
    @staticmethod
    def compute(relevances: list[bool], K: int, R: int):
        dcg = 0

        idcg = 0
        for rank in range(1, min(K, R) + 1):
            idcg += 1 / math.log2(rank + 1)
        
        for rank, relevance in enumerate(relevances, start=1):
            
            if rank > K:
                break

            if relevance:
                dcg += 1 / math.log2(rank + 1)
                break
            
        return dcg / idcg
