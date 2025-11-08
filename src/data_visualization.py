import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_preprocessing import (circle_data, 
                                diagonal_left_data,
                                diagonal_right_data,
                                horizontal_data,
                                vertical_data)

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
    
    # same scaling for each axis
    max_range = max(max(xs) - min(xs), 
                    max(ys) - min(ys), 
                    max(zs) - min(zs)) / 2
    
    mid_x = (max(xs) + min(xs)) / 2
    mid_y = (max(ys) + min(ys)) / 2
    mid_z = (max(zs) + min(zs)) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # display the data
    plt.show()


for i in range(len(horizontal_data)):
    visualize_movement(horizontal_data[i])