import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from data_preprocessing import (circle_data,
                                diagonal_left_data,
                                diagonal_right_data,
                                horizontal_data,
                                vertical_data)

# same scaling for each axis through finding the largest value of all movements
all_movements_data = (
    circle_data +
    diagonal_left_data +
    diagonal_right_data +
    horizontal_data +
    vertical_data
)
all_coords = [coord for movement in all_movements_data for coord in movement]

# find the largest value from all coordinates and scale the axis a little bit because augmented data can get bigger
MAX_VALUE = max(max(abs(coord[0]), abs(coord[1]), abs(coord[2])) for coord in all_coords) * 1.2

# visualize a movement from a coordinate list
def visualize_movement(coords_list, title="Movement"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # extract x, y, z
    xs = [coord[0] for coord in coords_list]
    ys = [coord[1] for coord in coords_list]
    zs = [coord[2] for coord in coords_list]
    
    # plot the trajectory
    ax.plot(xs, ys, zs, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)


    # set the axis limits
    ax.set_xlim(-MAX_VALUE, MAX_VALUE)
    ax.set_ylim(-MAX_VALUE, MAX_VALUE)
    ax.set_zlim(-MAX_VALUE, MAX_VALUE)

    # make the plot cubic
    ax.set_box_aspect([1, 1, 1])
    
    # display the data
    plt.show()

if __name__ == "__main__":
    # visualization tests can be inserted here
    for i in range(len(all_movements_data)):
        visualize_movement(all_movements_data[i])