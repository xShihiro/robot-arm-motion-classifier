"""Extract features from movement data for model training and evaluation."""

import math
from typing import Sequence

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

Movement = list[tuple[int, int, int]]
FeatureVector = list[float]

# Feature names in the same order as the feature vector assembled below
FEATURE_NAMES = [
    "total_x_movement",
    "total_y_movement",
    "total_z_movement",
    "path_length",
    "ratio_x_over_y",
    "ratio_x_over_z",
    "ratio_y_over_z",
    "frac_x_of_total",
    "frac_y_of_total",
    "frac_z_of_total",
    "axis_dx",
    "axis_dy",
    "axis_dz",
    "axis_length",
    "axis_dir_x",
    "axis_dir_y",
    "axis_dir_z",
    "peak_dx_from_start",
    "peak_dy_from_start",
    "peak_dz_from_start",
    "loopiness",
    "radius_std",
    "bbox_extent_x",
    "bbox_extent_y",
    "bbox_extent_z",
    "bbox_ratio_x_over_y",
    "bbox_ratio_x_over_z",
    "bbox_ratio_y_over_z",
    "frac_x_minus_y",
    "frac_x_minus_z",
    "frac_y_minus_z",
    "dir_weighted_xy",
    "dir_weighted_xz",
    "dir_weighted_yz",
    "peak_dx_normalized",
    "peak_dy_normalized",
    "peak_dz_normalized",
    "axis_dx_sq",
    "axis_dy_sq",
    "axis_dz_sq",
    "axis_dx_times_dz",
    "axis_dy_times_dz",
]


def _extract_total_movement(
    coord_list: Sequence[tuple[int, int, int]], coord_index: int
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
    point_a: tuple[int, int, int], point_b: tuple[int, int, int]
) -> int:
    """Return the squared distance between two 3D points."""
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    dz = point_a[2] - point_b[2]
    return dx * dx + dy * dy + dz * dz


def _find_extreme_points(
    movement: Sequence[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
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


def _find_furthest_point(movement: Sequence[tuple[int, int, int]], axis: int) -> int:
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


def extract_features(data_list: Sequence[Movement]) -> list[FeatureVector]:
    """Return a list of feature vectors for the provided movement sequences."""
    features: list[FeatureVector] = []
    for movement in data_list:
        # Total movement per axis
        total_x = _extract_total_movement(movement, 0)
        total_y = _extract_total_movement(movement, 1)
        total_z = _extract_total_movement(movement, 2)

        # Ratios between axes
        xy_ratio = _safe_ratio(total_x, total_y)
        xz_ratio = _safe_ratio(total_x, total_z)
        yz_ratio = _safe_ratio(total_y, total_z)

        # Share of total movement per axis
        total_sum = total_x + total_y + total_z
        frac_x = _safe_ratio(total_x, total_sum)
        frac_y = _safe_ratio(total_y, total_sum)
        frac_z = _safe_ratio(total_z, total_sum)

        # Longest axis between the two farthest points
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

        # The whole path length
        path_length = 0.0
        for i in range(len(movement) - 1):
            segment_dx = movement[i + 1][0] - movement[i][0]
            segment_dy = movement[i + 1][1] - movement[i][1]
            segment_dz = movement[i + 1][2] - movement[i][2]
            path_length += math.sqrt(segment_dx**2 + segment_dy**2 + segment_dz**2)

        # epsilon to avoid things like divisions by zero
        eps = 1e-6

        # Loopiness (path length vs axis length), high for circles, lower for straight-ish motions
        loopiness = path_length / (axis_length + eps)

        # all values per coord
        x_vals = [p[0] for p in movement]
        y_vals = [p[1] for p in movement]
        z_vals = [p[2] for p in movement]

        # approximate center of the trajectory
        center_x = sum(x_vals) / len(movement)
        center_y = sum(y_vals) / len(movement)
        center_z = sum(z_vals) / len(movement)

        # distances of all points from the center
        radii = []
        for x, y, z in movement:
            dx_c = x - center_x
            dy_c = y - center_y
            dz_c = z - center_z
            radii.append(math.sqrt(dx_c * dx_c + dy_c * dy_c + dz_c * dz_c))

        # radius consistency: low for circular trajectories, higher for line-ish paths
        mean_radius = sum(radii) / len(radii)
        radius_var = sum((r - mean_radius) ** 2 for r in radii) / len(radii)
        radius_std = math.sqrt(radius_var + eps)

        # Bounding box aspect ratios
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
            # Total movement per axis
            total_x,
            total_y,
            total_z,
            # length of the path
            path_length,
            # Ratios between axes
            xy_ratio,
            xz_ratio,
            yz_ratio,
            # Share of total movement per axis
            frac_x,
            frac_y,
            frac_z,
            # axis lengths between extreme points
            axis_dx,
            axis_dy,
            axis_dz,
            # total axis length between extreme points
            axis_length,
            # normalized axis directions
            axis_ndx,
            axis_ndy,
            axis_ndz,
            # peak displacements along each axis from the starting point
            peak_dx,
            peak_dy,
            peak_dz,
            # circle vs line geometry
            loopiness,
            radius_std,
            # bounding boxes of every axis and their ratios
            bbox_x,
            bbox_y,
            bbox_z,
            bbox_xy_ratio,
            bbox_xz_ratio,
            bbox_yz_ratio,
            # per-axis movement balance
            frac_x_minus_y,
            frac_x_minus_z,
            frac_y_minus_z,
            # direction-weighted fractions
            dir_xy,
            dir_xz,
            dir_yz,
            # normalized peaks
            peak_dx_norm,
            peak_dy_norm,
            peak_dz_norm,
            # squared axis displacements
            axis_dx**2,
            axis_dy**2,
            axis_dz**2,
            # cross-axis correlation between X/Y and Z (diagonals)
            (axis_dx * axis_dz),
            (axis_dy * axis_dz),
        ]

        features.append(feature_vector)

    return features


def prepare_all_data(augment: bool = True, n_augmentations: int = 3) -> tuple[
    list[FeatureVector],
    list[FeatureVector],
    list[FeatureVector],
    list[str],
    list[str],
    list[str],
]:
    """
    Split the data into training (50%), development (25%), and test (25%) sets,
    optionally augment, and extract features.
    """
    load_dataset()

    all_movement_data: list[Movement] = (
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
        train_size=0.5,
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

    X_train = extract_features(movements_train)
    X_dev = extract_features(movements_dev)
    X_test = extract_features(movements_test)

    return X_train, X_dev, X_test, labels_train, labels_dev, labels_test
