import random
import numpy as np
from data_preprocessing import horizontal_data
from data_visualization import visualize_movement

def gaussian_noise_movement(movement: list[tuple[int]], sigma=2.0) -> list[tuple[int]]:
    augmented = []

    for (x, y, z) in movement:
        nx = int(round(x + np.random.normal(0, sigma)))
        ny = int(round(y + np.random.normal(0, sigma)))
        nz = int(round(z + np.random.normal(0, sigma)))

        augmented.append((nx, ny, nz))

    return augmented

def time_warp_movement(movement: list[tuple[int]], strength=0.15) -> list[tuple[int]]:
    length = len(movement)
    warped = []
    i = 0

    while i < length:
        warped.append(movement[i])
        r = random.uniform(0, 1)

        # skip step (speed up) — probability = strength/2
        if r < strength / 2 and i < length - 2:
            i += 2   # sauter un point

        # duplicate step (slow down) — probability = strength/2
        elif r < strength:
            warped.append(movement[i])  # dupliquer un point
            i += 1

        # normal step
        else:
            i += 1

    return warped

# scale the movement isotropically or anisotropically by a random factor within a margin
def scale_movement(movement: list[tuple[int]], isotropical=True, margin=0.2) -> list[tuple[int]]:
    augmented_movement = []

    # the scale factor is chosen randomly in the margin to not get too unrealistic
    if isotropical:
        scale_factor_x = scale_factor_y = scale_factor_z = random.uniform(1 - margin, 1 + margin)

    # the scale factor is chosen randomly in the margin to stay in the right class
    else:
        scale_factor_x = random.uniform(1 - margin, 1 + margin)
        scale_factor_y = random.uniform(1 - margin, 1 + margin)
        scale_factor_z = random.uniform(1 - margin, 1 + margin)

    # scale each coordinate by the scale factor
    for coord in movement:
        augmented_movement.append((int(round(coord[0] * scale_factor_x)),
                                    int(round(coord[1] * scale_factor_y)),
                                    int(round(coord[2] * scale_factor_z))))

    return augmented_movement

def jitter_movement(movement: list[tuple[int]], max_jitter_distance: int = 10, jitter_prob: float = 0.25) -> list[tuple[int]]:
    augmented = []

    # jitter randomly in 25% of the coordinates by the jitter distance
    for x, y, z in movement:
        if random.random() < jitter_prob:
            jx = random.randint(-max_jitter_distance, max_jitter_distance)
            jy = random.randint(-max_jitter_distance, max_jitter_distance)
            jz = random.randint(-max_jitter_distance, max_jitter_distance)

            augmented.append((x + jx, y + jy, z + jz))
        # just keeps the original coordinate if not jittering
        else:
            augmented.append((x, y, z))

    return augmented

# augment a movement by a random transformation
def augment_movement(movement: list[tuple[int]]) -> list[tuple[int]]:
    rand = random.uniform(0, 1)

    if rand < 0.1:
        return scale_movement(movement, isotropical=True, margin=0.3)

    if rand < 0.2:
        return scale_movement(movement, isotropical=True, margin=0.15)

    elif rand < 0.4:
        return scale_movement(movement, isotropical=False)

    elif rand < 0.5:
        return jitter_movement(movement, max_jitter_distance=5, jitter_prob=0.15)

    elif rand < 0.6:
        return jitter_movement(movement, max_jitter_distance=7, jitter_prob=0.25)

    elif rand < 0.7:
        return gaussian_noise_movement(movement, sigma=2.0)

    elif rand < 0.8:
        return gaussian_noise_movement(movement, sigma=3.0)

    elif rand < 0.9:
        return time_warp_movement(movement, strength=0.3)

    else:
        return time_warp_movement(movement, strength=0.15)
