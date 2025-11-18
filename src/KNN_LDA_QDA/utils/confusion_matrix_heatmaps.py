# confusion_matrix_heatmaps.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrices(
    conf_mats: dict[str, dict[str, np.ndarray]],
    class_names: list[str]
) -> None:
    """
    For each model, create one figure with 3 heatmaps:
      Train, Dev, Test.

    Parameters
    ----------
    conf_mats : dict
        {
          "KNN": {
              "train": cm_train (C x C),
              "dev":   cm_dev,
              "test":  cm_test
          },
          "LDA": {...},
          ...
        }
    class_names : list of str
        Class label names in the same order as used by confusion_matrix.
    """
    split_order = ["train", "dev", "test"]
    split_titles = {"train": "Train", "dev": "Dev", "test": "Test"}

    for model_name, splits in conf_mats.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{model_name} - Confusion Matrices", fontsize=14)

        for i, split in enumerate(split_order):
            ax = axes[i]
            cm = splits.get(split, None)
            if cm is None:
                ax.axis("off")
                ax.set_title(f"{split_titles[split]} (no data)")
                continue

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax
            )
            ax.set_title(f"{split_titles[split]} set")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        plt.tight_layout()
        # leave a bit of space for suptitle
        plt.subplots_adjust(top=0.85)
        plt.show()
