import os
from os import path
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
from collections import Counter
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# input_sav_file = r"D:\Courses\CS419\project\cranfield\processed_data\term_idf.sav"
# input_csv_file = r"D:\Courses\CS419\project\cranfield\processed_data\document-term_matrix_tfidf_normalized.csv"
# input_query_path = r"D:\Courses\CS419\project\cranfield\queries"
# input_query_res_file = r"D:\Courses\CS419\project\cranfield\cranqrel"


input_sav_file = path.relpath(r"cranfield\processed_data\term_idf.sav")
input_csv_file = path.relpath(r"cranfield\processed_data\document-term_matrix_tfidf_normalized.csv")
input_query_path = path.relpath(r"cranfield\queries")
input_query_res_file = path.relpath(r"cranfield\cranqrel")


# current_dir = os.getcwd()
# path_in_common = os.path.commonpath([current_dir, input_sav_file])
# current_dir = current_dir.replace(path_in_common, "")

# input_sav_file = current_dir + input_sav_file
# input_csv_file = current_dir + input_csv_file
# input_query_path = current_dir + input_query_path
# input_query_res_file = current_dir + input_query_res_file


file = open(input_sav_file, 'rb')
terms_idf = pickle.load(file)
file.close()

document_term_matrix = pd.read_csv(input_csv_file, index_col=0)

with open(input_query_res_file, 'r', encoding='UTF-8') as f:
    lines = f.read().splitlines()

def get_text_from_file(filename):
    with open(filename, encoding='UTF-8', mode='r') as f:
        text = f.read()
    return text

# preprocess the query

def preprocess_text(text):
    processed_text = text.lower()
    processed_text = processed_text.replace("’", " ")
    processed_text = processed_text.replace("'s", " ")
    processed_text = processed_text.replace("'", " ")
    processed_text = processed_text.replace("“", '"')
    processed_text = processed_text.replace("”", '"')
    non_words = re.compile(r"[^A-Za-z']+")
    processed_text = re.sub(non_words, ' ', processed_text)
    return processed_text

def get_words_from_text(text):
    ps = PorterStemmer()
    processed_text = preprocess_text(text)
    filtered_words = [word for word in processed_text.split() if word not in stopwords.words('english')]
    processed_words = []
    for i in range(len(filtered_words)):
        processed_words.append(ps.stem(filtered_words[i]))
    return processed_words

##

def get_sorted_res(query):

    # process the query

    query_term_frequency_dict = dict(Counter(get_words_from_text(query)))

    query_vector_dict = {}

    for i in list(query_term_frequency_dict.keys()):
        if i in list(terms_idf.keys()):
            query_vector_dict[i] = query_term_frequency_dict[i] * terms_idf[i]

    l = np.linalg.norm(list(query_vector_dict.values()))

    for i in list(query_vector_dict.keys()):
        query_vector_dict[i] = query_vector_dict[i] / l

    ##

    ## print results

    temp = document_term_matrix.to_dict(orient='index')

    res = []

    for i in list(temp.keys()):
        keys = set(list(query_vector_dict.keys())).intersection(list(temp[i].keys()))
        dot_product = sum(query_vector_dict[k] * temp[i][k] for k in keys)
        if dot_product != 0:
            res.append((i, dot_product))
    
    res = sorted(res, key=lambda tup: tup[1], reverse=True)

    res_only = [i[0] for i in res]

    return res, res_only

def calculate_AP(res_only, res_gold):

    precision = 0
    precision_list = []
    recall_list = []
    
    for idx, i in enumerate(res_only):
        if i in res_gold:
            precision += 1
            precision_list.append(float(precision)/(idx+1))
            recall_list.append(float(precision)/len(res_gold))
    
    temp = 0
    for i in range(len(precision_list)-1, -1, -1):
        if temp >= precision_list[i]:
            precision_list[i] = temp
        elif temp < precision_list[i]:
            temp = precision_list[i]

    precision_list_11 = []
    points = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for idx, i in enumerate(points):
        for j in range(len(recall_list)):
            if recall_list[j] >= i:
                precision_list_11.append(max(precision_list[j:]))
                break
    
    if len(precision_list_11) < 11:
        precision_list_11 += [0]*(11-len(precision_list_11))

    average_precision = sum(precision_list_11)/len(precision_list_11)

    return average_precision

def calculate_MAP():

    AP_list = []

    j = '1'

    res_gold = []
    res_gold_list = []

    for i in lines:
        temp = i.split()
        if temp[0] == j:
            res_gold.append(int(temp[1]))
        elif temp[0] == '.':
            res_gold_list.append(res_gold)
            break
        else:
            j = temp[0]
            res_gold_list.append(res_gold)
            res_gold = []
            res_gold.append(int(temp[1]))
    res_only_list = []
    query_paths = sorted(os.listdir(input_query_path), key=lambda x: int(path.splitext( path.split(x)[1] )[0]))
    
    bar_format = "{l_bar}{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    with tqdm(range(len(query_paths)), desc="Calculating MAP", unit="query", bar_format=bar_format) as pbar:
        for query_file in query_paths:
            # print(query_file)
            pbar.set_postfix({'Processing': query_file})
            filename = os.path.join(input_query_path, query_file)
            res_only_list.append(get_sorted_res(get_text_from_file(filename))[1])
            pbar.update(1)
        
    for i in range(len(res_gold_list)):
        AP_list.append(calculate_AP(res_only_list[i],res_gold_list[i]))
    return sum(AP_list)/len(AP_list)

# print(calculate_MAP())

import time
start_time = time.time()
a = get_sorted_res("what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft . ")
print("Top 10 results for the query:")
print("\n".join([str(pairs) for pairs in a[0][:10]]))
print("--- mAP = %s ---" % (calculate_MAP()))
print("--- Execution time = %s seconds ---" % (time.time() - start_time))