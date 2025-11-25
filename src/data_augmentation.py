import math
import random
from typing import Callable, Dict, List, Tuple

import numpy as np

Movement = List[Tuple[int, int, int]]
Augmenter = Callable[[Movement], Movement]

DEFAULT_ISOTROPIC_MARGIN = 0.15
DEFAULT_ANISOTROPIC_MARGIN = 0.1
MAX_JITTER_DISTANCE = 10
JITTER_PROBABILITY = 0.4
CIRCLE_TRANSLATION_RANGE = 20


def _scale_movement(
    movement: Movement, isotropic: bool = True, margin: float = 0.2
) -> Movement:
    """Scale the movement isotropically or per axis inside the provided margin."""
    scale_factor_x = scale_factor_y = scale_factor_z = random.uniform(
        1 - margin, 1 + margin
    )
    if not isotropic:
        scale_factor_x = random.uniform(1 - margin, 1 + margin)
        scale_factor_y = random.uniform(1 - margin, 1 + margin)
        scale_factor_z = random.uniform(1 - margin, 1 + margin)

    augmented_movement: Movement = []
    for coord in movement:
        augmented_movement.append(
            (
                int(round(coord[0] * scale_factor_x)),
                int(round(coord[1] * scale_factor_y)),
                int(round(coord[2] * scale_factor_z)),
            )
        )

    return augmented_movement


def _jitter_movement(movement: Movement) -> Movement:
    """Jitter random coordinates inside the allowed maximum distance."""
    augmented_movement: Movement = []
    for coord in movement:
        if random.random() < JITTER_PROBABILITY:
            augmented_movement.append(
                (
                    coord[0]
                    + random.randint(-MAX_JITTER_DISTANCE, MAX_JITTER_DISTANCE),
                    coord[1]
                    + random.randint(-MAX_JITTER_DISTANCE, MAX_JITTER_DISTANCE),
                    coord[2]
                    + random.randint(-MAX_JITTER_DISTANCE, MAX_JITTER_DISTANCE),
                )
            )
        else:
            augmented_movement.append(coord)
    return augmented_movement


def _random_movement_transformation(
    movement: Movement, isotropic: bool, margin: float
) -> Movement:
    """Apply jitter/scale/both at random to diversify augmentation."""
    choice = random.random()
    augmented = movement
    if choice < 0.33:
        augmented = _jitter_movement(movement)
    elif choice < 0.66:
        augmented = _scale_movement(movement, isotropic=isotropic, margin=margin)
    else:
        augmented = _jitter_movement(movement)
        augmented = _scale_movement(augmented, isotropic=isotropic, margin=margin)
    return augmented


def _initialize_random_rotation_matrix() -> np.ndarray:
    """Return a 3x3 rotation matrix constructed from random Euler angles."""
    angle_x, angle_y, angle_z = np.random.uniform(0, 2 * math.pi, size=3)

    cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)
    cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
    cos_z, sin_z = math.cos(angle_z), math.sin(angle_z)

    rotation_x = np.array(
        [
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x],
        ]
    )

    rotation_y = np.array(
        [
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y],
        ]
    )

    rotation_z = np.array(
        [
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1],
        ]
    )

    return rotation_z @ rotation_y @ rotation_x


def _augment_circle(movement: Movement) -> Movement:
    """Apply a random 3D rotation (and occasional isotropic scaling) to preserve circular characteristics."""
    rotation_matrix = _initialize_random_rotation_matrix()
    coords = np.array(movement, dtype=float)
    rotated = coords @ rotation_matrix.T

    if random.random() < 0.5:
        rotated = _scale_movement(
            list(map(tuple, rotated.astype(int))),
            isotropic=True,
            margin=DEFAULT_ISOTROPIC_MARGIN,
        )
        rotated = np.array(rotated, dtype=float)

    return [
        (int(round(point[0])), int(round(point[1])), int(round(point[2])))
        for point in rotated
    ]


def _augment_diagonal_left(movement: Movement) -> Movement:
    return _random_movement_transformation(
        movement, isotropic=False, margin=DEFAULT_ANISOTROPIC_MARGIN
    )


def _augment_diagonal_right(movement: Movement) -> Movement:
    return _random_movement_transformation(
        movement, isotropic=False, margin=DEFAULT_ANISOTROPIC_MARGIN
    )


def _augment_horizontal(movement: Movement) -> Movement:
    return _random_movement_transformation(
        movement, isotropic=True, margin=DEFAULT_ISOTROPIC_MARGIN / 2
    )


def _augment_vertical(movement: Movement) -> Movement:
    return _random_movement_transformation(
        movement, isotropic=True, margin=DEFAULT_ISOTROPIC_MARGIN
    )


CLASS_AUGMENTERS: Dict[str, Augmenter] = {
    "circle": _augment_circle,
    "diagonal_left": _augment_diagonal_left,
    "diagonal_right": _augment_diagonal_right,
    "horizontal": _augment_horizontal,
    "vertical": _augment_vertical,
}


def augment_movement(movement: Movement, label: str) -> Movement:
    """Augment a movement using the class-specific strategy."""
    augmenter = CLASS_AUGMENTERS.get(label)
    return augmenter(movement)
