# Uses ONLY your provided feature_extraction.py + data_preprocessing.py
# No custom feature engineering is added.
import os
import sys

# Add parent folder (src/) to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PARENT_DIR)

# Now imports work
import data_preprocessing
import feature_extraction

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---- Your modules (unchanged) ----
from data_preprocessing import (
    circle_data,
    diagonal_left_data,
    diagonal_right_data,
    horizontal_data,
    vertical_data,
)
from feature_extraction import extract_features

# -----------------------------
# 1) Build dataset X, y
# -----------------------------
# Apply YOUR extractor to each class list; stack features; create labels.
X_list = []
y_list = []

for feats in extract_features(circle_data):
    X_list.append(feats); y_list.append("circle")

for feats in extract_features(diagonal_left_data):
    X_list.append(feats); y_list.append("diagonal_left")

for feats in extract_features(diagonal_right_data):
    X_list.append(feats); y_list.append("diagonal_right")

for feats in extract_features(horizontal_data):
    X_list.append(feats); y_list.append("horizontal")

for feats in extract_features(vertical_data):
    X_list.append(feats); y_list.append("vertical")

X = np.array(X_list, dtype=float)
y = np.array(y_list, dtype=str)

print(f"Loaded: {X.shape[0]} samples | feature dim = {X.shape[1]}")
print("Class counts:", {c: sum(y==c) for c in np.unique(y)})

# -----------------------------
# 2) Define models
# -----------------------------
knn = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")),
])

lda = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LDA(solver="svd")),
])

qda = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", QDA(reg_param=1e-3)),  # tiny regularization for stability
])

ensemble = VotingClassifier(
    estimators=[("knn", knn), ("lda", lda), ("qda", qda)],
    voting="hard"
)

models = {
    "KNN": knn,
    "LDA": lda,
    "QDA": qda,
    "Ensemble(HardVote)": ensemble
}

# -----------------------------
# 3) Train/test split & eval
# -----------------------------

# TODO: Create a util class to be use to give the results for the different model combination
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

for name, model in models.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_te, y_pred))
    print("Classification Report:\n", classification_report(y_te, y_pred, zero_division=0))

# -----------------------------
# 4) 5-fold CV (overall comparison)
# -----------------------------
print("\n=== 5-fold Cross-Validation ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")

# -----------------------------
# 5) Weighted voting (parallel, but stronger than hard vote)
# -----------------------------

# compute weights from CV on the *training* split
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
weights = {}
for name, model in [('KNN', knn), ('LDA', lda), ('QDA', qda)]:
    scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='accuracy')
    weights[name] = scores.mean()
print("Model weights (CV mean acc):", weights)

def weighted_vote_predict(X):
    # fit on full training split
    knn.fit(X_tr, y_tr); lda.fit(X_tr, y_tr); qda.fit(X_tr, y_tr)
    proba_sum = 0
    # all 3 provide predict_proba
    for w, mdl in [(weights['KNN'], knn), (weights['LDA'], lda), (weights['QDA'], qda)]:
        P = mdl.predict_proba(X)
        proba_sum = proba_sum + w * P if isinstance(proba_sum, np.ndarray) else w * P
    y_hat_idx = proba_sum.argmax(axis=1)
    classes = knn.classes_  # same class order across models after fit
    return classes[y_hat_idx]

y_hat_weighted = weighted_vote_predict(X_te)
print("\n=== Weighted Vote ===")
print("Accuracy:", accuracy_score(y_te, y_hat_weighted))
print("Confusion Matrix:\n", confusion_matrix(y_te, y_hat_weighted))
print("Classification Report:\n", classification_report(y_te, y_hat_weighted, zero_division=0))

