from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from matching import flag, normalize_text

# --- LOADING DATA ---

BASE_DIR = Path(__file__).resolve().parent
results_dir = BASE_DIR / "results"
results_dir.mkdir(parents = True, exist_ok = True)

procurements = pd.read_csv(BASE_DIR/"data"/"procurement.csv")
emission_cat = pd.read_csv(BASE_DIR/"data"/"categories.csv")

category_mapping = dict(zip(emission_cat["category_id"], emission_cat["category_name"]))


# --- PREPROCESSING ---

procurements["clean_description"] = procurements["description"].fillna("").apply(normalize_text)
category_ref = emission_cat["category_description"].fillna("").apply(normalize_text)


# --- FITTING TF_IDF ON TRAINING DATA ---

# Splitting data into training and test. Using stratification so all categories are represented
X_train_text, X_test_text, y_train, y_test = train_test_split(procurements["clean_description"], 
procurements["true_category_id"], test_size=0.4, random_state = 42, stratify = procurements["true_category_id"])
print(f"\nTraining dataset size: {len(y_train)}\nTesting dataset size: {len(y_test)}")

# Extracting features from training data + category description text corpus
vectorizer = TfidfVectorizer()
fitting_text = pd.concat([X_train_text, category_ref], ignore_index=True)
vectorizer.fit(fitting_text)

X_train_tfidf = vectorizer.transform(X_train_text)

# Extracting features from test data 
X_test_tfidf = vectorizer.transform(X_test_text)


# --- RIDGE CLASSIFIER ---
# A supervised classifier benchmark for the TF-IDF + cosine similarity method.

clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
print("\n====================\n= Ridge Classifier =\n")
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
plt.savefig(results_dir / "confusion_matrix_ridge.png", dpi=300, bbox_inches="tight")
plt.show()

# Accuracy score
score_ridge = accuracy_score(y_test, pred)
print(f"Accuracy Score: {score_ridge:.1%}")


# --- COSINE SIMILARITY (+ TF-IDF) ---
# Lexical Matching approach. Will compute cosine similarity for test set 
# for a fair comparison to the RidgeClassifier method.

category_ref_tfidf = vectorizer.transform(category_ref)

# Applying cosine similarity to the test set
procurement_cos = cosine_similarity(X_test_tfidf, category_ref_tfidf)

# Selecting the emission codes and purchase IDs for test set
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
plt.savefig(results_dir / "confusion_matrix_cos.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n=====================\n= Cosine Similarity =\n")

# Flagging
mask = flag(procurement_cos)
review_rate = mask.mean()

# Zero-similarity rate
mask_zero = flag(procurement_cos, 0)
zero_sim_rate = mask_zero.mean()

# Accuracy scores
score_cos_pure = accuracy_score(y_test, pred_cos_id)

accepted_mask = ~mask
score_cos_accepted = (accuracy_score(y_test[accepted_mask], pred_cos_id[~mask])
    if accepted_mask.any() # accounting for edge case: all matches flagged
    else np.nan
)

print(f"Accuracy score: {score_cos_pure:.1%}")
print(f"Accuracy score of unflagged (accepted) matches: {score_cos_accepted:.1%} ("
    f"{accepted_mask.sum()}/{len(mask)} accepted predictions)")
print(f"Flagged: {review_rate:.1%} of the purchases")
print(f"Zero-similarity rate: {zero_sim_rate:.1%}\n")
