import nltk
import re
import os
from os import path
import pickle
from nltk.corpus import stopwords
import math
from nltk.stem import PorterStemmer
from nltk import wordnet
from collections import Counter
import pandas as pd
import json
from tqdm import tqdm, trange
import numpy as np

import time

ps = PorterStemmer()

# input_path = r"D:\Courses\CS419\project\cranfield\docs"
# output_csv_file = r"D:\Courses\CS419\project\cranfield\processed_data\document-term_matrix.csv"
# output_sav_file = r"D:\Courses\CS419\project\cranfield\processed_data\document-term_frequency.sav"


input_path = path.relpath(r"movies/data/mad2plot.json")
output_csv_file = path.relpath(r"movies\sublinear_tf\document-term_matrix.csv")
output_sav_file = path.relpath(r"movies\sublinear_tf\document-term_frequency.sav")

data_path = path.split(input_path)[0]
output_path = path.split(output_csv_file)[0]
os.makedirs( output_path, exist_ok=True)
# sublinear_tf

# Read Json
def json_to_dict(json_file):
    with open(json_file, 'r', encoding='UTF-8') as f:
        data = json.load(f)
    return data

# preprocess text

def clean_text(text):
    processed_text = text.lower()
    processed_text = processed_text.replace("’", " ")
    processed_text = processed_text.replace("'s", " ")
    processed_text = processed_text.replace("'", " ")
    processed_text = processed_text.replace("“", '"')
    processed_text = processed_text.replace("”", '"')
    non_words = re.compile(r"[^A-Za-z']+")
    processed_text = re.sub(non_words, ' ', processed_text)
    return processed_text

def preprocessing(text):
    processed_text = clean_text(text)
    # Tokenization and stopword removal and stemming
    res = [ps.stem(word) for word in processed_text.split() if word not in stopwords.words('english')]
    # processed_words = []
    # for i in range(len(filtered_words)):
    #     processed_words.append(ps.stem(filtered_words[i]))
    return res

##

# def get_text_from_file(filename):
#     with open(filename, encoding='UTF-8', mode='r') as f:
#         text = f.read()
#     return text

def raw_tf(doc_list, id_list):
    list_of_dict = []
    with trange(len(doc_list), desc="Indexing", unit="plot", ncols=150) as progress_bar:
        for i in range(len(doc_list)):
            list_of_dict.append(dict(Counter(preprocessing(doc_list[i]))))
            progress_bar.set_postfix({"Plot ID": id_list[i]})
            progress_bar.update(1)
            
    
    res_df = pd.DataFrame.from_dict(list_of_dict).fillna(0)
    res_df.index = id_list
    
    
    res_df.to_csv(output_csv_file)
    return list_of_dict


def sublinear_tf(doc_list, id_list):
    list_of_dict = []
    with trange(len(doc_list) + 1, desc="Indexing", unit="plot", ncols=150) as pbar:
        for i in range(len(doc_list)):
            list_of_dict.append(dict(Counter(preprocessing(doc_list[i]))))
            pbar.set_postfix({"Plot ID": id_list[i]})
            pbar.update(1)
            
        
        res_df = pd.DataFrame.from_dict(list_of_dict).fillna(0)
        res_df.index = id_list
        
        
        pbar.set_description_str("Applying sublinearity...")
        res_df[res_df > 0].transform(lambda x: 1 + np.log2(x)).fillna(0)
        res_list = res_df.to_dict('records')
        pbar.update(1)
        
    res_df.to_csv(output_csv_file)
    return res_list


def indexing():
    
    start_time = time.time()

    ## Read JSON file and convert to dictionary
    with open(input_path, 'r', encoding='UTF-8') as f:
        data_dict = json.load(f)
    
    dup_keys = ['3001', '3014', '3020', '3021', '3023', '3033', '3040', '3050', '3060', '3066']
    for key in dup_keys:
        del data_dict[key]
        
    ## Save a mapping of movie ID to movie name
    title_list = list(data_dict.keys())
    ## Find duplicate IDs
    ## Process
    
    movie_id_name = {}
        
    for title in title_list:
        underscore_index = title.find('_')
        if underscore_index == -1:
            movie_id_name.update({title: ""})
        else:
            movie_id_name.update({title[:underscore_index]: title[underscore_index:]})
    
    # print(len(list(movie_id_name.keys())), dup_tits)
    with open( path.join(data_path, 'movie_id_name.json'), 'w') as f:
        json.dump(movie_id_name, f, indent=4)
        
    ## Get list of plots from the dictionary
    ### PREPROCESSING
    plot_list = list(data_dict.values())
    movie_ids = list(movie_id_name.keys())
    # print(len(plot_list), len(movie_ids))
    # with trange(len(plot_list), desc="Indexing", unit="plot", ncols=150) as progress_bar:
    #     for i in range(len(plot_list)):
    #         doc_vectors_dict.append(dict(Counter(preprocessing(plot_list[i]))))
    #         progress_bar.set_postfix({"Plot ID": movie_ids[i]})
    #         progress_bar.update(1)
    
    # doc_vectors_dict = raw_tf(plot_list, movie_ids)
    doc_vectors_dict = sublinear_tf(plot_list, movie_ids)

    # doc_vectors_df = pd.DataFrame.from_dict(doc_vectors_dict).fillna(0)
    # # doc_vectors_df.index = doc_vectors_df.index + 1
    # doc_vectors_df.index = movie_ids
    with open(output_sav_file, 'wb') as file:
        pickle.dump(doc_vectors_dict, file)

    print("--- %s seconds ---" % (time.time() - start_time))
   
 
if __name__ == "__main__":
    indexing()