import pandas as pd
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent

procurements = pd.read_csv(BASE_DIR/"data"/"procurement.csv")
emission_cat = pd.read_csv(BASE_DIR/"data"/"categories.csv")

### PREPROCESSING DATASETS ###

def normalize_text(text):
    """ 
    Removing punctuation, transforming to lowercase in preparation of using TF-IDF on text
    """

    text = re.sub(r"[^\w\s]"," ",text) #remove non-words/non-whitespaces
    text = re.sub(r"\s+"," ",text) #remove repeated whitespace
    text = text.lower()
    return text.strip()


procurements["clean_description"] = procurements["description"].fillna("").apply(normalize_text)

print(procurements["clean_description"])