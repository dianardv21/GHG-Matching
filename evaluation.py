from time import time
import pandas as pd
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from matching import normalize_text, flag

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
### FITTING TF_IDF ON TRAINING DATA ###

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
### RIDGE CLASSIFIER ###
## A supervised classifier benchmark for the TF-IDF + cosine similarity method.

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

# Visualization
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
ax.set_title("RidgeClassifier - Procurement Categories")

plt.tight_layout()
plt.savefig("confusion_matrix_ridge.png", dpi=300, bbox_inches="tight")
plt.show()

# Accuracy score
score_ridge = accuracy_score(y_test, pred)
print(f"Accuracy Score: {score_ridge}")

##################################
### COSINE SIMILARITY (+ TF-IDF) ###
## Lexical Matching approach. Will fit cosine similarity to test set 
## for a fair comparison to the RidgeClassifier method.

# Tokenizing category_description and weighting using TF-IDF
category_ref = emission_cat["category_description"].fillna("").apply(normalize_text)
category_ref_tfidf = vectorizer.transform(category_ref)

# Applying cosine similarity to the test set
procurement_cos = cosine_similarity(X_test_tfidf, category_ref_tfidf)

# Selecting the emission codes
pred_cos_index = np.argmax(procurement_cos, axis = 1)
pred_cos_id = emission_cat["category_id"].to_numpy()[pred_cos_index]

# Visualizing the highest score predictions
fig, ax = plt.subplots(figsize=(12, 10))
labels = emission_cat["category_id"].to_numpy()
display_names = [category_mapping[label] for label in labels]
ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred_cos_id,
    labels=labels,
    display_labels=display_names,
    ax=ax,
    xticks_rotation=90
)
ax.set_title("Cosine Similarity - Procurement Categories")

plt.tight_layout()
plt.savefig("confusion_matrix_cos.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\n====\nCosine Similarity:\nPredictions: {pred_cos_id}\n True Values: {y_test}\n")
print(procurement_cos)

# Accuracy score
score_cos = accuracy_score(y_test, pred_cos_id)
print(f"Accuracy Score: {score_cos}")

# Note: If all cosine similarity scores for a procurement are zero,
# np.argmax() defaults to the first category (index 0).
# To avoid treating these arbitrary assignments as valid predictions,
# the following section flags such procurements for manual review.

# Extracting purchase IDs
purchase_id_test = procurements.loc[X_test_text.index, "purchase_id"].to_numpy()

# Flagging
mask = flag(procurement_cos)
flagged_procurement = purchase_id_test[mask]
print(f"\n\n\nFlagged:{mask.sum()/len(mask)*100}% of the purchases")

