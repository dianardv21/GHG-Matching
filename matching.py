import pandas as pd
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.linear_model import RidgeClassifier
from time import time


BASE_DIR = Path(__file__).resolve().parent

procurements = pd.read_csv(BASE_DIR/"data"/"procurement.csv")
emission_cat = pd.read_csv(BASE_DIR/"data"/"categories.csv")

category_mapping = dict(zip(emission_cat["category_id"], emission_cat["category_name"]))

##################################
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

##################################
### TF_IDF MODEL FOR EMISSION MATCHING ###

# Splitting data into training and test
X_train_text, X_test_text, y_train, y_test = train_test_split(procurements["clean_description"], 
procurements["true_category_id"], test_size=0.4, random_state = 42)

# Extracting features from training data (sparse)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train_text)

# Extracting features from test data 
X_test_tfidf = vectorizer.transform(X_test_text)

feature_names = vectorizer.get_feature_names_out()

##################################
### TF_IDF + RIDGE CLASSIFIER ###
## A supervised classifier benchmark for the TF-IDF + cosine similarity method

clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
print(f"Ridge Classifier:")
t0 = time()
clf.fit(X_train_tfidf, y_train)
train_time = time() - t0
print(f"Train time: {train_time:.3}s")

t0 = time()
pred = clf.predict(X_test_tfidf)
test_time = time() - t0
print(f"Test time:  {test_time:.3}s")

fig, ax = plt.subplots(figsize=(12, 10))
labels = clf.classes_
display_names = [category_mapping[label] for label in labels]
ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred,
    labels=labels,
    display_labels=display_names,
    ax=ax,
    xticks_rotation=90
)
ax.set_title("RidgeClassifier — Procurement Categories")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

score_ridge = accuracy_score(y_test, pred)
print(f"Accuracy Score: {score_ridge}")