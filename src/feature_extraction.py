import math
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

# Updated feature names to match the actual feature vector order below
FEATURE_NAMES = [
    "ratio_xy",
    "ratio_xz",
    "ratio_yz",
    "frac_x",       # total_x / (total_x + total_y + total_z)
    "frac_y",       # total_y / ...
    "frac_z",       # total_z / ...
    "axis_dx",
    "axis_dy",
    "axis_dz",
    "axis_length",
    "axis_ndx",
    "axis_ndy",
    "axis_ndz",
    "peak_dx",
    "peak_dy",
    "peak_dz",
]


def _extract_total_movement(
    coord_list: Sequence[Tuple[int, int, int]], coord_index: int
) -> float:
    """Calculate the total movement on one axis for a single movement list."""
    total = 0.0
    for i in range(len(coord_list) - 1):
        total += abs(coord_list[i + 1][coord_index] - coord_list[i][coord_index])
    return total


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return the ratio of 2 values, or 0.0 if denominator is zero."""
    return numerator / denominator if denominator else 0.0


def _squared_distance(
    point_a: Tuple[int, int, int], point_b: Tuple[int, int, int]
) -> int:
    """Return the squared distance between two 3D points."""
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    dz = point_a[2] - point_b[2]
    return dx * dx + dy * dy + dz * dz


def _find_extreme_points(
    movement: Sequence[Tuple[int, int, int]],
) -> tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Return two points within the movement that have the largest distance between them."""
    extreme_a = movement[0]
    extreme_b = movement[1]
    max_distance = _squared_distance(extreme_a, extreme_b)

    for i in range(len(movement)):
        for j in range(i + 1, len(movement)):
            distance = _squared_distance(movement[i], movement[j])
            if distance > max_distance:
                max_distance = distance
                extreme_a = movement[i]
                extreme_b = movement[j]

    return extreme_a, extreme_b


def _find_furthest_point(movement: Sequence[Tuple[int, int, int]], axis: int) -> int:
    """Return the index of the point where the given axis has the largest distance from the start."""
    start_coord = movement[0][axis]
    max_distance = abs(movement[1][axis] - start_coord)
    furthest_index = 1

    for i in range(2, len(movement)):
        distance = abs(movement[i][axis] - start_coord)
        if distance > max_distance:
            max_distance = distance
            furthest_index = i

    return furthest_index


def _extract_features(data_list: Sequence[Movement]) -> List[FeatureVector]:
    """Return a list of feature vectors for the provided movement sequences."""
    features: List[FeatureVector] = []
    for movement in data_list:
        # Total movement per axis
        total_x = _extract_total_movement(movement, 0)
        total_y = _extract_total_movement(movement, 1)
        total_z = _extract_total_movement(movement, 2)

        # Ratios between axes (already used)
        xy_ratio = _safe_ratio(total_x, total_y)
        xz_ratio = _safe_ratio(total_x, total_z)
        yz_ratio = _safe_ratio(total_y, total_z)

        # NEW: share of total movement per axis
        total_sum = total_x + total_y + total_z
        frac_x = _safe_ratio(total_x, total_sum)
        frac_y = _safe_ratio(total_y, total_sum)
        frac_z = _safe_ratio(total_z, total_sum)

        # Longest axis between any two points (global “direction” of movement)
        axis_start, axis_end = _find_extreme_points(movement)
        axis_dx = axis_end[0] - axis_start[0]
        axis_dy = axis_end[1] - axis_start[1]
        axis_dz = axis_end[2] - axis_start[2]
        axis_length = math.sqrt(axis_dx**2 + axis_dy**2 + axis_dz**2)

        # Normalized axis direction
        axis_ndx = _safe_ratio(axis_dx, axis_length)
        axis_ndy = _safe_ratio(axis_dy, axis_length)
        axis_ndz = _safe_ratio(axis_dz, axis_length)

        # Peak displacement along each axis from the starting point
        peak_index_x = _find_furthest_point(movement, 0)
        peak_index_y = _find_furthest_point(movement, 1)
        peak_index_z = _find_furthest_point(movement, 2)
        peak_dx = movement[peak_index_x][0] - movement[0][0]
        peak_dy = movement[peak_index_y][1] - movement[0][1]
        peak_dz = movement[peak_index_z][2] - movement[0][2]

        eps = 1e-6

        # Loopiness (path length vs net displacement)
        net_disp = abs(axis_dx) + abs(axis_dy) + abs(axis_dz) + eps
        loopiness = axis_length / net_disp

        # Bounding box aspect ratios (assuming bbox_x, bbox_y, bbox_z exist)
        x_vals = [p[0] for p in movement]
        y_vals = [p[1] for p in movement]
        z_vals = [p[2] for p in movement]

        bbox_x = max(x_vals) - min(x_vals)
        bbox_y = max(y_vals) - min(y_vals)
        bbox_z = max(z_vals) - min(z_vals)

        bbox_xy_ratio = bbox_x / (bbox_y + eps)
        bbox_xz_ratio = bbox_x / (bbox_z + eps)
        bbox_yz_ratio = bbox_y / (bbox_z + eps)

        # Fraction differences
        frac_x_minus_y = frac_x - frac_y
        frac_x_minus_z = frac_x - frac_z
        frac_y_minus_z = frac_y - frac_z

        # Directional combinations
        dir_xy = axis_ndx * frac_x
        dir_xz = axis_ndx * frac_z
        dir_yz = axis_ndy * frac_z

        # Normalized peaks
        peak_dx_norm = peak_dx / (abs(axis_dx) + eps)
        peak_dy_norm = peak_dy / (abs(axis_dy) + eps)
        peak_dz_norm = peak_dz / (abs(axis_dz) + eps)

        feature_vector = [
            # --- original features ---
            xy_ratio,
            xz_ratio,
            yz_ratio,
            frac_x,
            frac_y,
            frac_z,
            axis_dx,
            axis_dy,
            axis_dz,
            axis_length,
            axis_ndx,
            axis_ndy,
            axis_ndz,
            peak_dx,
            peak_dy,
            peak_dz,

            # --- NEW: circle vs line geometry ---
            loopiness,  # HIGH for circles, lower for straight-ish motions

            # --- NEW: bounding box & shape ---
            bbox_x,
            bbox_y,
            bbox_z,
            bbox_xy_ratio,
            bbox_xz_ratio,
            bbox_yz_ratio,

            # --- NEW: per-axis movement balance ---
            frac_x_minus_y,
            frac_x_minus_z,
            frac_y_minus_z,

            # --- NEW: direction-weighted fractions ---
            dir_xy,
            dir_xz,
            dir_yz,

            # --- NEW: normalized peaks ---
            peak_dx_norm,
            peak_dy_norm,
            peak_dz_norm,

            # You can comment these in/out to test:
            # (Keep total length + loopiness, maybe drop some raw axes to reduce collinearity)

            axis_dx ** 2,
            axis_dy ** 2,
            axis_dz ** 2,
            (axis_dx * axis_dz),   # correlation between X and Z (diagonals)
            (axis_dy * axis_dz),
        ]

        features.append(feature_vector)

    return features


def prepare_all_data(
    augment: bool = True, n_augmentations: int = 3
) -> tuple[list, list, list, list, list, list]:
    """
    Split the data into training (30%), development (35%), and test (35%) sets,
    optionally augment, and extract features.
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
            label = labels_train[i]
            for _ in range(n_augmentations):
                movements_train.append(augment_movement(movements_train[i], label))
                labels_train.append(label)

    X_train = _extract_features(movements_train)
    X_dev = _extract_features(movements_dev)
    X_test = _extract_features(movements_test)

    return X_train, X_dev, X_test, labels_train, labels_dev, labels_test
