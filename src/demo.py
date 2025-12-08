"""Train a RandomForest on the combined dataset and predict unlabeled demo data."""

from collections import Counter
from pathlib import Path
from typing import List, Tuple, Any

from sklearn.ensemble import RandomForestClassifier

from data_augmentation import augment_movement
from data_preprocessing import (
    DATA_DIRECTORY,
    PROJECT_ROOT,
    circle_data,
    diagonal_left_data,
    diagonal_right_data,
    horizontal_data,
    load_dataset,
    movement_into_tuple_list,
    vertical_data,
)
from feature_extraction import FEATURE_NAMES, extract_features
from random_forest import RF_CONFIG

Movement = List[Tuple[int, int, int]]

# important: insert Demo data directory name in the string
DEMO_DIRECTORY: Path = PROJECT_ROOT / "fake_demo_data"

# Demo config assembled from random_forest settings
DEMO_CONFIG: dict[str, Any] = {
    "augment": RF_CONFIG["augment"],
    "n_augmentations": RF_CONFIG["n_augmentations"],
    "model_params": RF_CONFIG["model_params"],
}


def _load_training_data(
    train_dir: Path, augment: bool, n_augmentations: int
) -> tuple[list[Movement], list[str]]:
    """Use existing preprocessing to fill the five class arrays and build training sets."""
    load_dataset(train_dir)

    movements = (
        circle_data
        + diagonal_left_data
        + diagonal_right_data
        + horizontal_data
        + vertical_data
    )
    labels: list[str] = (
        ["circle"] * len(circle_data)
        + ["diagonal_left"] * len(diagonal_left_data)
        + ["diagonal_right"] * len(diagonal_right_data)
        + ["horizontal"] * len(horizontal_data)
        + ["vertical"] * len(vertical_data)
    )

    if augment:
        original_len = len(movements)
        for idx in range(original_len):
            label = labels[idx]
            for _ in range(n_augmentations):
                movements.append(augment_movement(movements[idx], label))
                labels.append(label)

    return movements, labels


def _load_demo_movements(demo_dir: Path) -> tuple[list[Movement], list[str]]:
    """Load all demo text files (unlabeled) into a movement list."""
    demo_movements: list[Movement] = []
    demo_files: list[str] = []

    for file in sorted(demo_dir.iterdir()):
        if file.suffix != ".txt":
            continue
        demo_movements.append(movement_into_tuple_list(str(file)))
        demo_files.append(file.name)

    return demo_movements, demo_files


def run_demo(train_dir: Path, demo_dir: Path) -> None:
    """Train on labeled data and predict the unlabeled demo set."""
    print(f"\nLoading and augmenting training data ...")
    train_movements, train_labels = _load_training_data(
        train_dir, DEMO_CONFIG["augment"], DEMO_CONFIG["n_augmentations"]
    )
    X_train = extract_features(train_movements)

    print(f"Training samples created: {len(X_train)}")

    print("\nTraining Random Forest model ...")
    rf = RandomForestClassifier(**DEMO_CONFIG["model_params"])
    rf.fit(X_train, train_labels)
    print("Model trained.")

    print("\nRandom Forest feature importances:", rf.feature_importances_)
    print("\nFeature order used:")
    print(", ".join(FEATURE_NAMES))

    print(f"\nLoading demo data ...")
    demo_movements, demo_files = _load_demo_movements(demo_dir)
    X_demo = extract_features(demo_movements)
    print(f"{len(X_demo)} demo samples loaded.")

    predictions = rf.predict(X_demo)
    counts = Counter(predictions)

    print("\nPrediction counts:")
    for label, count in counts.most_common():
        print(f"  {label}: {count}")

    print("\nPredictions per file:")
    for file_ref, pred in zip(demo_files, predictions):
        print(f"  {file_ref}: {pred}")


if __name__ == "__main__":
    run_demo(DATA_DIRECTORY, DEMO_DIRECTORY)
