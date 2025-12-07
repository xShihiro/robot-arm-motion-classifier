import matplotlib.pyplot as plt
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from feature_extraction import FEATURE_NAMES, prepare_all_data

# Random Forest Hyperparameter Grid for Grid Search
RF_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7, 9, None],
    "min_samples_leaf": [1, 2, 3, 4],
    "min_samples_split": [2, 4, 6],
    "max_features": ["sqrt", "log2"],
}

# Default params (e.g. for demo/predict scripts) from a good prior grid-search result
RF_DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 7,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "random_state": 8,
    "n_jobs": -1,
}

# Central configuration dict
RF_CONFIG: dict[str, Any] = {
    "augment": True,
    "n_augmentations": 30,
    "cv_folds": 5,
    "visualize_tree": False,
    "tree_index_to_show": 0,
    "evaluate_test_set": True,
    "param_grid": RF_PARAM_GRID,
    "model_params": RF_DEFAULT_MODEL_PARAMS,
}


def main() -> None:
    """Train the random forest, evaluate on the dev set,
    and optionally visualize a tree and evaluate on the test set."""
    X_train, X_dev, X_test, y_train, y_dev, y_test = prepare_all_data(
        augment=RF_CONFIG["augment"], n_augmentations=RF_CONFIG["n_augmentations"]
    )

    print(f"\nAmount of Data used for Training: {len(X_train)}")
    print(f"\nAmount of Data in Development Set: {len(X_dev)}")
    print(f"\nAmount of Data in Test Set: {len(X_test)}")

    # Random Forest with Hyperparameter Search
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RF_CONFIG["model_params"]["random_state"]),
        RF_CONFIG["param_grid"],
        cv=RF_CONFIG["cv_folds"],
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

    if RF_CONFIG["visualize_tree"]:
        tree_to_plot = rf.estimators_[RF_CONFIG["tree_index_to_show"]]

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
        plt.title(f"Random Forest — Tree #{RF_CONFIG['tree_index_to_show']}")
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
    if RF_CONFIG["evaluate_test_set"]:
        y_test_pred_rf = rf.predict(X_test)
        test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
        print(f"\n[RandomForest] Accuracy on test set: {test_accuracy_rf:.2f}")
        print("\n[RandomForest] Test Set Detailed classification report:")
        print(classification_report(y_dev, y_dev_pred_rf))
        print("\n[RandomForest] Test Set Confusion matrix:")
        print(confusion_matrix(y_test, y_test_pred_rf))


if __name__ == "__main__":
    main()
