import random
from data_preprocessing import horizontal_data
from data_visualization import visualize_movement

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

def jitter_movement(movement: list[tuple[int]]) -> list[tuple[int]]:
    augmented_movement = []

    # the amount in mm the robot arm can jitter has a set maximum distance
    max_jitter_distance = 10

    # jitter randomly in 40% of the coordinates by the jitter distance
    for coord in movement:
        if random.random() < 0.4:
            augmented_movement.append((coord[0] + random.randint(-max_jitter_distance, max_jitter_distance),
                                        coord[1] + random.randint(-max_jitter_distance, max_jitter_distance),
                                        coord[2] + random.randint(-max_jitter_distance, max_jitter_distance)))

        # just keeps the original coordinate if not jittering
        else:
            augmented_movement.append(coord)

    return augmented_movement

# augment a movement by a random transformation
def augment_movement(movement: list[tuple[int]]) -> list[tuple[int]]:
    rand = random.random()

    if rand < 0.5:
        return jitter_movement(movement)
    elif rand < 0.8:
        return scale_movement(movement, isotropical=True, margin=0.15)
    else:
        return scale_movement(movement, isotropical=False, margin=0.1)