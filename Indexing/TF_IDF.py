import json
#from tfidf_sublinearIDF_0.tfidf_index.def_tfidf import TFIDFPipeline
from Indexing.tfidf_baseline_1.tfidf_index.def_tfidf import TFIDFPipeline
#tfidf_baseline_1.tfidf_index.def_tfidf import TFIDFPipeline
#from Indexing.TF_IDF import 
#from BERT import SemanticChunker

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

if __name__ == "__main__":

    with open('Datasets\\imdb2plot.json', encoding="utf8") as f:
        imdb2plot = json.load(f)

    with open('Datasets\\imdb2title.json', encoding="utf8") as f:
        imdb2title = json.load(f)


    # mad2plot_data = {}
    # for key in imdb2title:
    #     if imdb2plot.get(key) is not None:
    #         mad2plot_data[imdb2title[key]] = imdb2plot[key]

    # pipeline = TFIDFPipeline(mad2plot_data)
    # pipeline("aaaa")
    pipeline = TFIDFPipeline.load_local("Indexing\\tfidf_sublinearIDF_0\\tfidf_index")
    result = pipeline("Hello")
    #print(result)

    retrieved_titles = []
    for title, score, docs in result:
        retrieved_titles.append(title)

    print(retrieved_titles[:10])
    #print(docs[:10])

    pass