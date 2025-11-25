# train_dev_test_accuracy_plot.py
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_curves(accuracies: dict[str, dict[str, float]]) -> None:
    """
    Plot grouped bars for Train / Dev / Test accuracy per model.

    Parameters
    ----------
    accuracies : dict
        {
          "KNN": {"train": ..., "dev": ..., "test": ...},
          "LDA": {...},
          ...
        }
    """
    models = list(accuracies.keys())
    x = np.arange(len(models))
    width = 0.25

    train_vals = [accuracies[m]["train"] for m in models]
    dev_vals = [accuracies[m]["dev"] for m in models]
    test_vals = [accuracies[m]["test"] for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, train_vals, width, label="Train")
    ax.bar(x,         dev_vals,   width, label="Dev")
    ax.bar(x + width, test_vals,  width, label="Test")

    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs Dev vs Test Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.show()
