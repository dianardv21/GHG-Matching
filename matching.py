import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def normalize_text(text):
    """ 
    Normalize text for TF-IDF vectorization.

    Removing punctuation, transforming to lowercase in preparation of using TF-IDF on text.

    Input: 
        text: String to normalize
    Output: 
        Normalized string
    """

    text = re.sub(r"[^\w\s]"," ",text) #remove non-words/non-whitespaces
    text = re.sub(r"\s+"," ",text) #remove repeated whitespace
    text = text.lower()
    return text.strip()


def flag(procurement_cos, threshold = 0.25):
    """
    Flag procurement descriptions with zero or low maximum cosine similarity scores for manual review.
    Input:  
        procurement_cos: 2D array of cosine similarity scores (rows = procurements, columns = emission categories)
        threshold: Maximum similarity score at/below which a procurement is flagged for review (Default = 0.25)
    Output: 
        mask: Boolean mask array indicating which procurements need review
    """

    max_cos = np.max(procurement_cos, axis=1)
    mask = max_cos <= threshold

    return mask


### MAIN ###

if __name__ == "__main__": 

    ##################################
    ### LOADING DATA ###
    # Note: Current procurement data for this project is very limited (~50 entries), 
    # real-world datasets would have a lot more entries, providing a broader vocabulary
    # and more reliable TF-IDF weights. 

    BASE_DIR = Path(__file__).resolve().parent

    procurements = pd.read_csv(BASE_DIR/"data"/"procurement.csv")
    emission_cat = pd.read_csv(BASE_DIR/"data"/"categories.csv")

    category_mapping = dict(zip(emission_cat["category_id"], emission_cat["category_name"]))

    ##################################
    ### PREPROCESSING DATASETS ###

    procurements["clean_description"] = procurements["description"].fillna("").apply(normalize_text)

    ##################################
    ### APPLYING TF-IDF ###



    ##################################
    ### FLAGGING ###

    # Extracting purchase IDs
    purchase_id_test = procurements.loc[procurement_indices, purchase_IDs_column].to_numpy()
    mask = flag(procurement_cos)
    flagged_procurement = purchase_id_test[mask]



