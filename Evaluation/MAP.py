import json

class AP_K:
    @staticmethod
    def compute(relevance_values: list[bool]):

        precision_sum = 0

        rel_count = 0
        for rank, rel_value in enumerate(relevance_values, start=1):
            if not rel_value:
                continue
            
            rel_count += 1
            precision_sum += rel_count / rank

        if rel_count == 0:
            return 0
        
        return precision_sum / rel_count
    
class MAP:
    @staticmethod
    def compute(multiple_relevance_values: list[list[bool]]):

        avg_precision_sum = 0

        for relevance_values in multiple_relevance_values:
            ap_k = AP_K.compute(relevance_values)
            avg_precision_sum += ap_k
        
        return avg_precision_sum / len(multiple_relevance_values)
    
class Ranked2Relevance:
    @staticmethod
    def convert(ranked_result: dict):

        multiple_relevance_values = []

        for truth_doc in ranked_result.keys():
            multiple_relevance_docs = ranked_result[truth_doc]

            for relevance_docs in multiple_relevance_docs:
                relevance_values = [True if doc == truth_doc else False for doc in relevance_docs]
                multiple_relevance_values.append(relevance_values)
                
        return multiple_relevance_values

if __name__ == "__main__":

    output_json_path = "Output/BM25.json"
    with open(output_json_path, 'r') as f:
        ranked_result = json.load(f)

        multiple_relevance_values = Ranked2Relevance.convert(ranked_result)
        
        map = MAP.compute(multiple_relevance_values)
        print(map)

    pass