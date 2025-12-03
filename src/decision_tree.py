from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from feature_extraction import FEATURE_NAMES, prepare_all_data

SHOW_TREE = False
EVALUATE_TEST_SET = True
AUGMENT = True
# getting similar results w/ n_aug = 2, CV_folds = 5 or 10 (test acc fluctuates between ~0.7 - ~0.85)
N_AUGMENTATIONS = 3
CV_FOLDS = 10
PARAM_GRID = {
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
    "min_samples_leaf": [2, 3, 4, 5, 6],
    "min_samples_split": [2, 4, 6, 8, 10],
}

# Random Forest hyperparameters
USE_RANDOM_FOREST = True
RF_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7, None],
    "min_samples_leaf": [1, 2, 3],
    "min_samples_split": [2, 4, 6],
}


def main():
    """Train the decision tree, evaluate on the dev set, and optionally display diagnostics."""
    X_train, X_dev, X_test, y_train, y_dev, y_test = prepare_all_data(
        augment=AUGMENT, n_augmentations=N_AUGMENTATIONS
    )

    print(f"\nAmount of Data used for Training: {len(X_train)}")
    print(f"\nAmount of Data in Development Set: {len(X_dev)}")
    print(f"\nAmount of Data in Test Set: {len(X_test)}")

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=8),
        PARAM_GRID,
        cv=CV_FOLDS,
        scoring="accuracy",
    )
    grid.fit(X_train, y_train)

    print(
        "\nBest hyperparameters found (DecisionTree):",
        grid.best_params_,
        "with score",
        grid.best_score_,
    )

    dtc = grid.best_estimator_
    print("\nDecision Tree feature importances:", dtc.feature_importances_)

    y_dev_pred = dtc.predict(X_dev)
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    print(f"\n[DecisionTree] Accuracy on development set: {dev_accuracy:.2f}")
    print("\n[DecisionTree] Detailed classification report:")
    print(classification_report(y_dev, y_dev_pred))
    print("\n[DecisionTree] Confusion matrix:")
    print(confusion_matrix(y_dev, y_dev_pred))

    if SHOW_TREE:
        feature_names = FEATURE_NAMES[: len(X_train[0])]
        plt.figure(figsize=(20, 10))
        plot_tree(
            dtc,
            feature_names=feature_names,
            class_names=[
                "circle",
                "diagonal_left",
                "diagonal_right",
                "horizontal",
                "vertical",
            ],
            filled=True,
            fontsize=10,
        )
        plt.show()

    if EVALUATE_TEST_SET:
        y_test_pred = dtc.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"\n[DecisionTree] Accuracy on test set: {test_accuracy:.2f}")
        print("\n[DecisionTree] Confusion matrix:")
        print(confusion_matrix(y_test, y_test_pred))

    # ---------------- Random Forest ----------------
    if USE_RANDOM_FOREST:
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

        rf = rf_grid.best_estimator_
        print("\nRandom Forest feature importances:", rf.feature_importances_)

        # Dev eval
        y_dev_pred_rf = rf.predict(X_dev)
        dev_accuracy_rf = accuracy_score(y_dev, y_dev_pred_rf)
        print(f"\n[RandomForest] Accuracy on development set: {dev_accuracy_rf:.2f}")
        print("\n[RandomForest] Detailed classification report:")
        print(classification_report(y_dev, y_dev_pred_rf))
        print("\n[RandomForest] Confusion matrix:")
        print(confusion_matrix(y_dev, y_dev_pred_rf))

        # Test eval
        if EVALUATE_TEST_SET:
            y_test_pred_rf = rf.predict(X_test)
            test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
            print(f"\n[RandomForest] Accuracy on test set: {test_accuracy_rf:.2f}")
            print("\n[RandomForest] Confusion matrix:")
            print(confusion_matrix(y_test, y_test_pred_rf))


if __name__ == "__main__":
    main()
