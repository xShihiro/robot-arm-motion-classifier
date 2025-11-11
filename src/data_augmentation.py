import random

from numpy.ma.core import max_val

from data_preprocessing import horizontal_data
from data_visualization import visualize_movement

# scale the movement isotropically by a random factor within a range
def scale_movement(movement: list[tuple[int]]) -> list[tuple[int]]:
    augemented_movement = []

    # the scale factor is chosen randomly between 0.5 and 2
    scale_factor = random.uniform(0.5, 2)

    # scale each coordinate by the scale factor
    for coord in movement:
        augemented_movement.append((int(round(coord[0] * scale_factor)),
                                    int(round(coord[1] * scale_factor)),
                                    int(round(coord[2] * scale_factor))))

    return augemented_movement

def jitter_movement(movement: list[tuple[int]]) -> list[tuple[int]]:
    augemented_movement = []

    # the amount in mm the robot arm can jitter has a set maximum distance
    max_jitter_distance = 5

    # jitter randomly in 25% of the coordinates by the jitter distance
    for coord in movement:
        isjittering = random.uniform(0, 1) < 0.25
        if isjittering:
            augemented_movement.append((coord[0] + random.randint(-max_jitter_distance, max_jitter_distance),
                                        coord[1] + random.randint(-max_jitter_distance, max_jitter_distance),
                                        coord[2] + random.randint(-max_jitter_distance, max_jitter_distance)))

        # just keeps the original coordinate if not jittering
        else:
            augemented_movement.append(coord)
    
    return augemented_movement