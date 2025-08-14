import os
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
from collections import Counter
import pickle
import pandas as pd
import numpy as np
from indexing import preprocessing
from tqdm import tqdm, trange
import time
import json

# input_file = r"movies\processed_data\document-term_matrix_tfidf_normalized.csv"
input_sav_file = r"movies\sublinear_tf\term_idf.sav"
input_csv_file = r"movies\sublinear_tf\document-term_matrix_tfidf_normalized.csv"
id_name_path = r"movies\data\movie_id_name.json"
queries_path = r"movies\annotation\queries.json"
output_infer = r"movies\batch_infer\output_infer.json"

def dot_product(query_vector, corpus_matrix):
    """
    Calculate the dot product between a query vector and a normalized corpus.
    
    Parameters:
    query_vector (pd.DataFrame): The query vector.
    norm_corpus (pd.DataFrame): The normalized corpus.
    
    Returns:
    dict: A dictionary with document IDs as keys and their corresponding dot products as values.
    """
    
    expaned_query_vect = corpus_matrix.merge(query_vector, how='right').fillna(0)
    query_matrix = pd.concat( [expaned_query_vect.loc[[0]]] * corpus_matrix.shape[0] )
    query_matrix.index = corpus_matrix.index
    
    dot_product = (query_matrix * corpus_matrix).sum(axis=1)
    res = dot_product.sort_values(ascending=False)
    # res = np.flip(dot_product.sort(axis=-1))
    # print(dot_product)
    return res.to_dict()

def cosine_similarity(query_vector, corpus_matrix):
    """
    Calculate the cosine similarity between a query vector and a normalized corpus.
    
    Parameters:
    query_vector (pd.DataFrame): The query vector.
    norm_corpus (pd.DataFrame): The normalized corpus.
    
    Returns:
    dict: A dictionary with document IDs as keys and their corresponding cosine similarities as values.
    """
    
    expaned_query_vect = corpus_matrix.merge(query_vector, how='right').fillna(0)
    query_matrix = pd.concat( [expaned_query_vect.loc[[0]]] * corpus_matrix.shape[0] )
    query_matrix.index = corpus_matrix.index
    
    dot_product = (query_matrix * corpus_matrix).sum(axis=1)
    
    # norm_query = np.linalg.norm(query_vector.values)
    # norm_corpus = np.linalg.norm(corpus_matrix.values, axis=1)
    
    norm_query = query_matrix.pow(2).sum(axis=1).apply(np.sqrt)
    norm_corpus = corpus_matrix.pow(2).sum(axis=1).apply(np.sqrt)
    
    # print(dot_product)
    # print(norm_query)
    # print(norm_corpus)  
    
    # print(np.array(expaned_query_vect.pow(2)).sum(axis=1))
    
    # exit()
    
    denominator = norm_query * norm_corpus
    denominator.index = dot_product.index
    cos_sim = dot_product / denominator
    res = cos_sim.sort_values(ascending=False)
    
    # print(dot_product)
    # print(denominator)
    # print(cos_sim.values)
    # print(res.size)
    
    return res.to_dict()


def inference(query, top_k=10, score_func="experimental"):
    with open(input_sav_file, 'rb') as f:
        terms_idf = pickle.load(f)
    
    query_term_frequency_dict = dict(Counter(preprocessing(query)))

    # query_tfidf_dict = {}
    
    query_tfidf = pd.DataFrame.from_dict([query_term_frequency_dict]).fillna(0)

    ## Calculate & Normalize TF_IDF
    for term in query_tfidf:
        if term in list(terms_idf.keys()):
            query_tfidf[term] = query_tfidf[term] * terms_idf[term]

    l = np.linalg.norm(list(query_tfidf.loc[0]))
    norm_query = query_tfidf.copy() / l

    norm_corpus = pd.read_csv(input_csv_file, index_col=0)
    
    terms_intersection = norm_query.columns.intersection(norm_corpus.columns)
    norm_query = norm_query[terms_intersection]
    norm_corpus = norm_corpus[terms_intersection]
    
    ### Calculation on pandas
    # pd_time = time.time()
    
    # bar_format = "{l_bar}{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    # with tqdm(total=4, unit='step', bar_format=bar_format) as pbar:
    if score_func == "dot_product":
        res = dot_product(norm_query, norm_corpus)
        # top_k_res = list(res.items())[:top_k]
        # print(  "\n".join([str(ans) for ans in top_k_res])   )
    elif score_func == "cosine_similarity":
        res = cosine_similarity(norm_query, norm_corpus)
        # top_k_res = list(res.items())[:top_k]
        # print(  "\n".join([str(ans) for ans in top_k_res])   )
    else:
        res_dot = dot_product(norm_query, norm_corpus)
        res_cos = cosine_similarity(norm_query, norm_corpus)
        k_dot = list(res_dot.items())[:top_k]
        k_cos = list(res_cos.items())[:top_k]
        res = pd.DataFrame([k_dot, k_cos],
                           index=["dot_product", "cosine_similarity"],
                           columns=np.arange(top_k)+1
                           ).T
        # print(res)
        
    return res
    
    # top_k_res = list(res.items())[:top_k]
    # print(  "\n".join([str(ans) for ans in top_k_res])   )
    # print(f"Time taken using pandas: {time.time() - pd_time} seconds")
    ###


def batch_inference(top_k=5, score_func='cosine_similarity'):
    
    with open(id_name_path, "r") as f:
        id_name_dic = json.load(f)
    with open(queries_path, "r") as f:
        queries_dic = json.load(f)
    
    # def get_title(movie_id):
        
    with trange(len(queries_dic.keys()), desc="Batch Infernce", unit="query", ncols=150) as p_bar:
        res = {}
        for movie in queries_dic:
            title_ret = []
            for query in queries_dic[movie]:
                doc_rel = inference(query, top_k, score_func)
                top_doc = list(doc_rel.keys())[:top_k]
                # top_id = ["{doc:04d}" for doc in top_doc if str(doc)<4 else str(doc)]
                top_id = list(map(lambda x: f"{x:04d}", top_doc))
                top_title = [doc_id + id_name_dic[doc_id] for doc_id in top_id]
                title_ret.append(top_title)
            res[movie] = title_ret
            p_bar.set_postfix({"Movie": movie})
            p_bar.update(1)
    
    return res
    
if __name__ == "__main__":
    
    top_k = 5
    score_function = "cosine_similarity"

    with open(id_name_path, "r") as f:
        id_name_dic = json.load(f)
    ### plot summary about 200 words
    # query = """Harry, shaken by the Ministry incident, hesitates to return to Hogwarts, but Dumbledore convinces him and takes him to meet Slughorn, hoping he’ll return to teach. Meanwhile, Death Eaters cause chaos, including kidnapping Ollivander and destroying his shop. Bellatrix makes Snape swear an Unbreakable Vow to protect Draco and complete his mission if he fails. At Hogwarts, Slughorn teaches Potions and Snape teaches Defense Against the Dark Arts. Harry uses an old Potions book full of notes by the "Half-Blood Prince" and excels. Ron becomes Quidditch goalie and dates Lavender, upsetting Hermione. Harry is jealous of Ginny’s closeness with Dean.

    # At Christmas, Death Eaters burn the Burrow. Dumbledore shows Harry memories of Tom Riddle, one of which is false. Using a Luck potion, Harry gets the real memory from Slughorn, learning about Horcruxes. Dumbledore reveals two Horcruxes and tells Harry they must find the rest.

    # Harry suspects Draco of planning something. After Ron is poisoned, Harry uses a spell from the Prince’s book on Draco, badly injuring him. Ginny hides the book. Later, Harry and Dumbledore retrieve another Horcrux. At Hogwarts, Draco lets Death Eaters in, and Snape kills Dumbledore. The Horcrux is fake. Harry, Ron, and Hermione decide to hunt the others together.
    # """

    ### no proper nouns, summarized at 100 words
    # query = "A boy, shaken by recent events, is urged by an elder to return to school and meet a former teacher. Meanwhile, dark followers cause destruction and plot with a young student tasked with a secret mission. At school, the boy excels using an old book with mysterious notes. He uncovers a method the villain used to hide parts of his soul in objects. After a failed attempt to save the headmaster, who is killed by a man under magical oath, the boy learns the retrieved object is fake. He and his two friends vow to find and destroy the real ones."

    # query = "A young student returns to school, urged by a wise mentor. A former teacher joins the staff, while dark followers create chaos and a secret mission is given to another student. The boy discovers a powerful book filled with notes, helping him excel. He learns that the enemy split their soul into several objects to gain immortality. With his mentor, he retrieves one such object, only to find it is fake. The mentor is killed by a man who made a magical vow. The student, joined by two close friends, resolves to track down the real objects and end the threat."
    
    # query = "After a harrowing experience, a student hesitates to return but is convinced by an older figure. At school, he finds help in a new teacher and a mysterious book. Meanwhile, dark forces make moves, and a secretive student works on a mission involving a magical device. The boy retrieves a soul-containing item with the elder, but it turns out to be fake. The elder is betrayed and killed by a trusted figure. Grieving but determined, the boy, along with two loyal friends, decides not to return to school next year and instead to seek and destroy the remaining soul-bound objects."
    # print("Searching...")
    # res = inference(query, score_func=score_function)
    # ### Minh họa
    # top_k_res = list(res.items())[:top_k]
    # print(  "\n".join([str(ans) for ans in top_k_res])   )
    # ### Minh họa
    
    # print(list(id_name_dic.items())[:top_k])
    # print(list(res.keys())[:top_k])
    
    
    output_dict = batch_inference(top_k=5, score_func='cosine_similarity')
    
    os.makedirs(r"movies\batch_infer", exist_ok=True)
    with open(output_infer, "w") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)
    
    ### L2 norm của 1 vector với các phần tử rất nhỏ (xấp xỉ 0) == 1 nên cosine ở trường hợp này xấp xỉ bằng tích vô hướng