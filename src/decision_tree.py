from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from feature_extraction import FEATURE_NAMES, prepare_all_data

SHOW_TREE = True
EVALUATE_TEST_SET = False
AUGMENT = True
N_AUGMENTATIONS = 3
CV_FOLDS = 5
PARAM_GRID = {
    "max_depth": [5, 6, 7, 8, 9, 10],
    "min_samples_leaf": [2, 3, 4, 5, 6],
    "min_samples_split": [4, 6, 8, 10],
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

    print("\nGrid search results:", grid.cv_results_)

    print(
        "\nBest hyperparameters found:",
        grid.best_params_,
        "with score",
        grid.best_score_,
    )

    dtc = grid.best_estimator_
    print("\nFeature importances:", dtc.feature_importances_)

    y_dev_pred = dtc.predict(X_dev)

    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    print(f"\nAccuracy on development set: {dev_accuracy:.2f}")

    print("\nDetailed classification report:")
    print(classification_report(y_dev, y_dev_pred))

    print("\nConfusion matrix:")
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
        print(f"\nAccuracy on test set: {test_accuracy:.2f}")
        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_test_pred))


if __name__ == "__main__":
    main()
