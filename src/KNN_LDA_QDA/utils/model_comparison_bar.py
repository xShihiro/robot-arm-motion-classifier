# model_comparison_bar.py
import matplotlib.pyplot as plt


def plot_model_comparison_bar(test_accuracies: dict[str, float]) -> None:
    """
    Plot a bar chart of test accuracies for each model.

    Parameters
    ----------
    test_accuracies : dict
        Mapping model_name -> test_accuracy (float).
    """
    models = list(test_accuracies.keys())
    vals = [test_accuracies[m] for m in models]

    plt.figure(figsize=(10, 5))
    plt.bar(models, vals)
    plt.ylabel("Test Accuracy")
    plt.title("Model Performance Comparison (Test Set)")
    plt.ylim(0, 1.1)

    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.show()
