from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

from feature_extraction import prepare_all_data

SHOW_TREE = True
EVALUATE_TEST_SET = False


def main():
    """Train the decision tree, evaluate on the dev set, and optionally display diagnostics."""
    X_train, X_dev, X_test, y_train, y_dev, y_test = prepare_all_data(
        augment=True, n_augmentations=3
    )

    print(f"\nAmount of Data used for Training: {len(X_train)}")
    print(f"\nAmount of Data in Development Set: {len(X_dev)}")
    print(f"\nAmount of Data in Test Set: {len(X_test)}")

    dtc = DecisionTreeClassifier(random_state=8)
    dtc.fit(X_train, y_train)

    y_dev_pred = dtc.predict(X_dev)

    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    print(f"\nAccuracy on development set: {dev_accuracy:.2f}")

    print("\nDetailed classification report:")
    print(classification_report(y_dev, y_dev_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_dev, y_dev_pred))

    if SHOW_TREE:
        plt.figure(figsize=(20, 10))
        plot_tree(
            dtc,
            feature_names=[
                "total_x",
                "total_y",
                "total_z",
                "ratio_xy",
                "ratio_xz",
                "ratio_yz",
            ],
            class_names=[
                "circle",
                "diagonal_left",
                "diagonal_right",
                "horizontal",
                "vertical",
            ],
            filled=True,
            fontsize=10,
            max_depth=3,
        )
        plt.show()

    if EVALUATE_TEST_SET:
        y_test_pred = dtc.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"\nAccuracy on test set: {test_accuracy:.2f}")


if __name__ == "__main__":
    main()
