import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_THRESHOLD = 0.1


def normalize_text(text):
    """ 
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


def flag(procurement_cos, threshold = DEFAULT_THRESHOLD):
    """
    Flag procurement descriptions with zero or low maximum cosine similarity scores for manual review.
    Input:  
        procurement_cos: 2D array of cosine similarity scores (rows = procurements, columns = emission categories)
        threshold: Maximum similarity score at/below which a procurement is flagged for review (Default = 0.1)
    Output: 
        mask: Boolean mask array indicating which procurements need review
    """

    max_cos = np.max(procurement_cos, axis=1)
    mask = max_cos <= threshold

    return mask


# --- MAIN ---

if __name__ == "__main__": 

    # --- LOADING DATA ---
    # Note: Current procurement data for this project is very limited (~50 entries), 
    # real-world datasets would have a lot more entries, providing a broader vocabulary
    # and more reliable TF-IDF weights. 

    BASE_DIR = Path(__file__).resolve().parent

    procurements = pd.read_csv(BASE_DIR/"data"/"procurement.csv")
    emission_cat = pd.read_csv(BASE_DIR/"data"/"categories.csv")

    category_mapping = dict(zip(emission_cat["category_id"], emission_cat["category_name"]))


    # --- PREPROCESSING DATASETS ---

    procurements["clean_description"] = procurements["description"].fillna("").apply(normalize_text)
    category_ref = emission_cat["category_description"].fillna("").apply(normalize_text)


    # --- APPLYING TF-IDF ---

    vectorizer = TfidfVectorizer()
    fitting_text = pd.concat([procurements["clean_description"], category_ref], ignore_index=True)
    vectorizer.fit(fitting_text)

    procurements_tfidf = vectorizer.transform(procurements["clean_description"])
    category_ref_tfidf = vectorizer.transform(category_ref)

    
    # --- CALCULATING COSINE SIMILARITY ---

    procurement_cos = cosine_similarity(procurements_tfidf, category_ref_tfidf)
    pred_cos_index = np.argmax(procurement_cos, axis = 1)
    pred_cos_id = emission_cat["category_id"].to_numpy()[pred_cos_index]                               


    # --- FLAGGING ---

    # Applying flagging mask
    mask = flag(procurement_cos)
    print(f"\nFlagged: {mask.mean():.1%} of the purchases for a threshold of {DEFAULT_THRESHOLD}.\n")

    # --- GENERATING CSV ---
    # Output: a .csv file with procurements, assigned emission categories and flag 

    output = procurements[["purchase_id", "description"]].copy()
    output["emission_code"] = pred_cos_id
    output["emission_category"] = output["emission_code"].map(category_mapping)
    output["cosine_similarity"] = np.max(procurement_cos, axis = 1)
    output["marked_for_review"] = mask

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents = True, exist_ok = True)
    output.to_csv(results_dir /"predictions.csv", index = False)
