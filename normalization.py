import pandas as pd
import math
import pickle
import numpy as np
import os
from os import path
from tqdm import tqdm

input_csv_file = path.relpath(r"movies\sublinear_tf\document-term_matrix.csv")
input_sav_file = path.relpath(r"movies\sublinear_tf\term_idf.sav")
output_path = path.relpath(r"movies\sublinear_tf")

def cal_tf_idf(term_freq, terms_idf):
    for term in term_freq:
        term_freq[term] = term_freq[term].astype(float) * terms_idf[term]
    return term_freq

def normalize_tf_idf(tf_idf):
    bar_format = "{l_bar}{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    
    with tqdm(total=len(tf_idf.index), desc="Normalizing", unit='document', bar_format=bar_format) as pbar:
        for i in tf_idf.index:
            l = np.linalg.norm(list(tf_idf.loc[i]))
            # for term in tf_idf:
            tf_idf.loc[i] = tf_idf.loc[i] / l
            pbar.set_postfix({"document": i})
            pbar.update(1)
                
    return tf_idf

def normalization():
    document_term_matrix = pd.read_csv(input_csv_file, index_col=0)
    with open(input_sav_file, 'rb') as f:
        terms_idf = pickle.load(f)
        
    tf_idf_matrix = cal_tf_idf(document_term_matrix.copy(), terms_idf)

    # Normalization
    norm_tf_idf = normalize_tf_idf(tf_idf_matrix.copy())

    norm_tf_idf.to_csv(os.path.join(output_path, "document-term_matrix_tfidf_normalized.csv"))
    # print(document_term_matrix)

if __name__ == "__main__":
    normalization()