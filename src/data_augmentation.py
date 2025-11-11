import random
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
