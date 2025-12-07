from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from feature_extraction import FEATURE_NAMES, prepare_all_data

# Configuration
AUGMENT = True
N_AUGMENTATIONS = 30
CV_FOLDS = 5
VISUALIZE_TREE = False
TREE_INDEX_TO_SHOW = 0  # index of the tree in the forest to visualize
EVALUATE_TEST_SET = True

# Random Forest Hyperparameter Grid for Grid Search
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7, 9, None],
    "min_samples_leaf": [1, 2, 3, 4],
    "min_samples_split": [2, 4, 6],
    "max_features": ["sqrt", "log2"],
}


def main():
    """Train the decision tree, evaluate on the dev set, and optionally display diagnostics."""
    X_train, X_dev, X_test, y_train, y_dev, y_test = prepare_all_data(
        augment=AUGMENT, n_augmentations=N_AUGMENTATIONS
    )

    print(f"\nAmount of Data used for Training: {len(X_train)}")
    print(f"\nAmount of Data in Development Set: {len(X_dev)}")
    print(f"\nAmount of Data in Test Set: {len(X_test)}")

    # Random Forest with Hyperparameter Search
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=8),
        RF_PARAM_GRID,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)

    print(
        "\nBest hyperparameters found (RandomForest):",
        rf_grid.best_params_,
        "with score",
        rf_grid.best_score_,
    )

    # Train final model with best hyperparameters and display feature importances
    rf = rf_grid.best_estimator_
    print("\nRandom Forest feature importances:", rf.feature_importances_)

    if VISUALIZE_TREE:
        tree_to_plot = rf.estimators_[TREE_INDEX_TO_SHOW]

        plt.figure(figsize=(24, 12))
        plot_tree(
            tree_to_plot,
            feature_names=FEATURE_NAMES,
            class_names=[
                "circle",
                "diagonal_left",
                "diagonal_right",
                "horizontal",
                "vertical",
            ],
            filled=True,
            fontsize=8,
        )
        plt.title(f"Random Forest — Tree #{TREE_INDEX_TO_SHOW}")
        plt.show()

    # Dev eval
    y_dev_pred_rf = rf.predict(X_dev)
    dev_accuracy_rf = accuracy_score(y_dev, y_dev_pred_rf)
    print(f"\n[RandomForest] Accuracy on development set: {dev_accuracy_rf:.2f}")
    print("\n[RandomForest] Dev Set Detailed classification report:")
    print(classification_report(y_dev, y_dev_pred_rf))
    print("\n[RandomForest] Dev Set Confusion matrix:")
    print(confusion_matrix(y_dev, y_dev_pred_rf))

    # Test eval
    if EVALUATE_TEST_SET:
        y_test_pred_rf = rf.predict(X_test)
        test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
        print(f"\n[RandomForest] Accuracy on test set: {test_accuracy_rf:.2f}")
        print("\n[RandomForest] Test Set Detailed classification report:")
        print(classification_report(y_dev, y_dev_pred_rf))
        print("\n[RandomForest] Test Set Confusion matrix:")
        print(confusion_matrix(y_test, y_test_pred_rf))


if __name__ == "__main__":
    main()
