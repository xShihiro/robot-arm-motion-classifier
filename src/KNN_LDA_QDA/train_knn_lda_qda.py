# train_knn_lda_qda.py
# Uses ONLY your provided feature_extraction.py + data_preprocessing.py
# No custom feature engineering is added.
import os
import sys

# Add parent folder (src/) to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PARENT_DIR)

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from feature_extraction import prepare_all_data  # <-- uses your logic

# plotting helpers
from KNN_LDA_QDA.utils.confusion_matrix_heatmaps import plot_confusion_matrices
from KNN_LDA_QDA.utils.feature_scatter_3d import plot_feature_scatter_3d
from KNN_LDA_QDA.utils.model_comparison_bar import plot_model_comparison_bar
from KNN_LDA_QDA.utils.train_dev_test_accuracy_plot import plot_accuracy_curves



# -----------------------------
# 1) Build dataset via prepare_all_data
# -----------------------------
# This already:
#   - merges all classes,
#   - splits into train/dev/test (70/15/15 or your new logic),
#   - performs augmentation (if enabled),
#   - extracts features.
X_train, X_dev, X_test, y_train, y_dev, y_test = prepare_all_data(
    augment=True,
    n_augmentations=10
)

# Convert to numpy arrays for sklearn
X_train = np.array(X_train, dtype=float)
X_dev   = np.array(X_dev, dtype=float)
X_test  = np.array(X_test, dtype=float)

y_train = np.array(y_train, dtype=str)
y_dev   = np.array(y_dev, dtype=str)
y_test  = np.array(y_test, dtype=str)

print(f"Train: {X_train.shape[0]} samples | Dev: {X_dev.shape[0]} | Test: {X_test.shape[0]}")
print("Train class counts:", {c: sum(y_train == c) for c in np.unique(y_train)})


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

# Dictionaries to store metrics
train_accuracies: dict[str, float] = {}
dev_accuracies: dict[str, float] = {}
test_accuracies: dict[str, float] = {}

conf_mats: dict[str, dict[str, np.ndarray]] = {}


# -----------------------------
# 3) Train on TRAIN, evaluate on DEV and TEST
# -----------------------------
for name, model in models.items():
    model.fit(X_train, y_train)

    # --- TRAIN metrics ---
    y_pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    cm_train = confusion_matrix(y_train, y_pred_train)

    # --- DEV metrics ---
    y_pred_dev = model.predict(X_dev)
    acc_dev = accuracy_score(y_dev, y_pred_dev)
    cm_dev = confusion_matrix(y_dev, y_pred_dev)

    # --- TEST metrics ---
    y_pred_test = model.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    cm_test = confusion_matrix(y_test, y_pred_test)

    # Store in dicts
    train_accuracies[name] = acc_train
    dev_accuracies[name] = acc_dev
    test_accuracies[name] = acc_test

    conf_mats[name] = {
        "train": cm_train,
        "dev": cm_dev,
        "test": cm_test
    }

    # Print nicely as before
    print(f"\n=== {name} (Dev set) ===")
    print(f"Dev Accuracy: {acc_dev:.3f}")
    print("Dev Confusion Matrix:\n", cm_dev)
    print("Dev Classification Report:\n",
          classification_report(y_dev, y_pred_dev, zero_division=0))

    print(f"\n=== {name} (Test set) ===")
    print(f"Test Accuracy: {acc_test:.3f}")
    print("Test Confusion Matrix:\n", cm_test)
    print("Test Classification Report:\n",
          classification_report(y_test, y_pred_test, zero_division=0))


# -----------------------------
# 4) 5-fold CV on TRAIN (overall comparison)
# -----------------------------
print("\n=== 5-fold Cross-Validation on TRAIN ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")


# -----------------------------
# 5) Weighted voting using TRAIN performance
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
weights: dict[str, float] = {}
for name, model in [("KNN", knn), ("LDA", lda), ("QDA", qda)]:
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    weights[name] = scores.mean()
print("Model weights (CV mean acc):", weights)


def weighted_vote_predict(X: np.ndarray) -> np.ndarray:
    # fit on full training split
    knn.fit(X_train, y_train)
    lda.fit(X_train, y_train)
    qda.fit(X_train, y_train)

    proba_sum = None
    for w, mdl in [(weights["KNN"], knn),
                   (weights["LDA"], lda),
                   (weights["QDA"], qda)]:
        P = mdl.predict_proba(X)
        proba_sum = w * P if proba_sum is None else proba_sum + w * P

    y_hat_idx = proba_sum.argmax(axis=1)
    classes = knn.classes_  # same class order across models after fit
    return classes[y_hat_idx]


# --- Weighted vote metrics (train/dev/test) ---
y_hat_weighted_train = weighted_vote_predict(X_train)
y_hat_weighted_dev   = weighted_vote_predict(X_dev)
y_hat_weighted_test  = weighted_vote_predict(X_test)

acc_w_train = accuracy_score(y_train, y_hat_weighted_train)
acc_w_dev   = accuracy_score(y_dev,   y_hat_weighted_dev)
acc_w_test  = accuracy_score(y_test,  y_hat_weighted_test)

cm_w_train = confusion_matrix(y_train, y_hat_weighted_train)
cm_w_dev   = confusion_matrix(y_dev,   y_hat_weighted_dev)
cm_w_test  = confusion_matrix(y_test,  y_hat_weighted_test)

train_accuracies["WeightedVote"] = acc_w_train
dev_accuracies["WeightedVote"]   = acc_w_dev
test_accuracies["WeightedVote"]  = acc_w_test

conf_mats["WeightedVote"] = {
    "train": cm_w_train,
    "dev": cm_w_dev,
    "test": cm_w_test
}

print("\n=== Weighted Vote (Test set) ===")
print("Accuracy:", acc_w_test)
print("Confusion Matrix:\n", cm_w_test)
print("Classification Report:\n",
      classification_report(y_test, y_hat_weighted_test, zero_division=0))


# -----------------------------
# 6) Build accuracy dict for plotting
# -----------------------------
accuracy_dict: dict[str, dict[str, float]] = {}
for name in ["KNN", "LDA", "QDA", "Ensemble(HardVote)", "WeightedVote"]:
    accuracy_dict[name] = {
        "train": train_accuracies[name],
        "dev":   dev_accuracies[name],
        "test":  test_accuracies[name],
    }

# Class names (sorted so they match confusion_matrix default ordering)
all_labels = np.concatenate([y_train, y_dev, y_test])
class_names = sorted(list(np.unique(all_labels)))


# -----------------------------
# 7) Call plotting scripts
# -----------------------------

# Script 1: Test accuracy bar plot
plot_model_comparison_bar(test_accuracies=test_accuracies)

# Script 2: confusion matrices – 1 figure per model, 3 heatmaps per figure
plot_confusion_matrices(conf_mats, class_names)

# Script 3: 3D feature scatter (use all data)
X_all = np.vstack([X_train, X_dev, X_test])
y_all = np.concatenate([y_train, y_dev, y_test])
plot_feature_scatter_3d(X_all, y_all, title="3D Scatter of All Feature Vectors")

# Script 4: Train vs Dev vs Test accuracy per model
plot_accuracy_curves(accuracy_dict)
