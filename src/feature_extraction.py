from typing import List, Sequence, Tuple

from sklearn.model_selection import train_test_split

from data_augmentation import augment_movement

from data_preprocessing import load_dataset
from data_preprocessing import (
    circle_data,
    diagonal_left_data,
    diagonal_right_data,
    horizontal_data,
    vertical_data,
)

Movement = List[Tuple[int, int, int]]
FeatureVector = List[float]


def _extract_total_movement(
    coord_list: Sequence[Tuple[int, int, int]], coord_index: int
) -> float:
    """Calculate the total movement on one axis for a single movement list."""
    total = 0.0
    for i in range(len(coord_list) - 1):
        total += abs(coord_list[i + 1][coord_index] - coord_list[i][coord_index])
    return total


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _extract_features(data_list: Sequence[Movement]) -> List[FeatureVector]:
    """Return a list of feature vectors for the provided movement sequences."""
    features: List[FeatureVector] = []
    for movement in data_list:
        total_x = _extract_total_movement(movement, 0)
        total_y = _extract_total_movement(movement, 1)
        total_z = _extract_total_movement(movement, 2)

        feature_vector = [
            total_x,
            total_y,
            total_z,
            _safe_ratio(total_x, total_y),
            _safe_ratio(total_x, total_z),
            _safe_ratio(total_y, total_z),
        ]
        features.append(feature_vector)

    return features


def prepare_all_data(
    augment: bool = True, n_augmentations: int = 3
) -> tuple[list, list, list, list, list, list]:
    """
    Split the data into training (30%), development (35%), and test (35%) sets, optionally augment, and extract features.
    """
    load_dataset()

    all_movement_data: List[Movement] = (
        circle_data
        + diagonal_left_data
        + diagonal_right_data
        + horizontal_data
        + vertical_data
    )

    all_labels = (
        ["circle"] * len(circle_data)
        + ["diagonal_left"] * len(diagonal_left_data)
        + ["diagonal_right"] * len(diagonal_right_data)
        + ["horizontal"] * len(horizontal_data)
        + ["vertical"] * len(vertical_data)
    )

    movements_train, movements_temp, labels_train, labels_temp = train_test_split(
        all_movement_data,
        all_labels,
        train_size=0.3,
        stratify=all_labels,
        random_state=8,
        shuffle=True,
    )
    movements_dev, movements_test, labels_dev, labels_test = train_test_split(
        movements_temp,
        labels_temp,
        train_size=0.5,
        stratify=labels_temp,
        random_state=8,
        shuffle=True,
    )

    if augment:
        original_train_length = len(movements_train)
        for i in range(original_train_length):
            for _ in range(n_augmentations):
                movements_train.append(augment_movement(movements_train[i]))
                labels_train.append(labels_train[i])

    X_train = _extract_features(movements_train)
    X_dev = _extract_features(movements_dev)
    X_test = _extract_features(movements_test)

    return X_train, X_dev, X_test, labels_train, labels_dev, labels_test
