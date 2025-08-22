import pandas as pd
import numpy as np
import math
import pickle
import os
from os import path
import json
from tqdm import tqdm

input_file = path.relpath(r"movies\sublinear_tf\document-term_matrix.csv")
output_path = path.relpath(r"movies\sublinear_tf")

def calculate_idf(doc_term_csv_path, output_path):
    document_term_matrix = pd.read_csv(doc_term_csv_path, index_col=0, dtype=int)

    N_CORPUS = len(document_term_matrix.index)
    txt_file = ""

    idf = {}
    bar_format = "{l_bar}{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    with tqdm(total=len(document_term_matrix.columns), desc="Calculating IDF", unit='term', bar_format=bar_format) as pbar:
        for term in document_term_matrix.columns:
            # Count the number of documents containing the term
            n_doc = document_term_matrix[term].astype(bool).sum()  # Count non-zero entries for the term
            # if n > 0:
            term_idf = math.log(N_CORPUS / n_doc + 1)  # Calculate IDF
            txt_file += f"{term}: {term_idf}\n"
            pbar.set_postfix({"term": term, "idf": np.round(term_idf, 4)})
            idf[term] = term_idf
            pbar.update(1)

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "term_idf.txt"), 'w+') as f:
        f.write(txt_file)

    with open(os.path.join(output_path, "term_idf.sav"), 'wb') as f:
        pickle.dump(idf, f)
    
if __name__ == "__main__":
    calculate_idf(input_file, output_path)