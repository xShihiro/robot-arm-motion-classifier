# feature_scatter_3d.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)
import numpy as np


def plot_feature_scatter_3d(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "3D Scatter of Feature Vectors"
) -> None:
    """
    3D scatter plot of 3D feature vectors, colored by class label.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, 3). Feature vectors (total movement per axis).
    y : np.ndarray
        Shape (N,). Class labels.
    title : str
        Plot title.
    """
    if X.shape[1] != 3:
        raise ValueError(f"Expected X to have shape (N, 3), got {X.shape}")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    classes = np.unique(y)
    for cls in classes:
        idx = (y == cls)
        ax.scatter(
            X[idx, 0],
            X[idx, 1],
            X[idx, 2],
            label=str(cls),
            s=40
        )

    ax.set_xlabel("Movement X")
    ax.set_ylabel("Movement Y")
    ax.set_zlabel("Movement Z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
