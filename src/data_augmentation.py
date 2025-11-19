import random
from typing import List, Tuple

Movement = List[Tuple[int, int, int]]

DEFAULT_ISOTROPIC_MARGIN = 0.15
DEFAULT_ANISOTROPIC_MARGIN = 0.1
MAX_JITTER_DISTANCE = 10
JITTER_PROBABILITY = 0.4


def _scale_movement(movement: Movement, isotropic: bool = True, margin: float = 0.2) -> Movement:
    """Scale the movement isotropically or per axis inside the provided margin."""
    augmented_movement: Movement = []
    if isotropic:
        scale_factor_x = scale_factor_y = scale_factor_z = random.uniform(1 - margin, 1 + margin)
    else:
        scale_factor_x = random.uniform(1 - margin, 1 + margin)
        scale_factor_y = random.uniform(1 - margin, 1 + margin)
        scale_factor_z = random.uniform(1 - margin, 1 + margin)

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
                    coord[0] + random.randint(-MAX_JITTER_DISTANCE, MAX_JITTER_DISTANCE),
                    coord[1] + random.randint(-MAX_JITTER_DISTANCE, MAX_JITTER_DISTANCE),
                    coord[2] + random.randint(-MAX_JITTER_DISTANCE, MAX_JITTER_DISTANCE),
                )
            )
        else:
            augmented_movement.append(coord)
    return augmented_movement


def augment_movement(movement: Movement) -> Movement:
    """Augment a movement by applying a random jitter or scaling transformation."""
    rand = random.random()
    if rand < 0.5:
        return _jitter_movement(movement)
    if rand < 0.8:
        return _scale_movement(movement, isotropic=True, margin=DEFAULT_ISOTROPIC_MARGIN)
    return _scale_movement(movement, isotropic=False, margin=DEFAULT_ANISOTROPIC_MARGIN)
