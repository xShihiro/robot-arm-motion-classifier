from sklearn.model_selection import train_test_split
from data_augmentation import augment_movement
from data_preprocessing import (circle_data,
                                diagonal_left_data,
                                diagonal_right_data,
                                horizontal_data,
                                vertical_data)


# calculate the total movement on one axis from a movement list
def extract_total_movement(coord_list: tuple, coord_index: int) -> float:
    # float to mat
    res = 0.0
    
    # calculate the distances between one coordinate for every step and sum them up
    for i in range(len(coord_list)-1):
        res += abs(coord_list[i+1][coord_index] - coord_list[i][coord_index])
    
    # sum of all 1-step distances = total movement
    return res

# receive a list of movement data lists and return a list of feature vectors
def extract_features(data_list: list) -> list:
    res = []
    
    # iterate over every movement in data list and create a 3-dimension 
    # feature vector containing the total movement for each coordinate
    for movement in data_list:
        temp_total_movements = [extract_total_movement(movement, 0),
                                extract_total_movement(movement, 1),
                                extract_total_movement(movement, 2)]
        res.append(temp_total_movements)
    
    return res

# split the data into training (70%), development (15%) and test set (15%), augment it and extract features
def prepare_all_data(augment=True, n_augmentations=3) -> tuple[list, list, list, list, list, list]:
    # combine the raw movement data
    all_movement_data = (
        circle_data +
        diagonal_left_data +
        diagonal_right_data +
        horizontal_data +
        vertical_data
    )

    # create the label list
    all_labels = (
        ["circle"] * len(circle_data) +
        ["diagonal_left"] * len(diagonal_left_data) +
        ["diagonal_right"] * len(diagonal_right_data) +
        ["horizontal"] * len(horizontal_data) +
        ["vertical"] * len(vertical_data)
    )

    # split the data into training (70%), development (15%) and test set (15%)
    movements_train, movements_temp, labels_train, labels_temp = train_test_split(all_movement_data, all_labels, train_size=0.7, random_state=8, shuffle=True)
    movements_dev, movements_test, labels_dev, labels_test = train_test_split(movements_temp, labels_temp, train_size=0.5, random_state=8, shuffle=True)

    # augment the training data n times and append the corresponding labels
    if augment:
        original_train_length = len(movements_train)
        for i in range(original_train_length):
            for j in range(n_augmentations):
                movements_train.append(augment_movement(movements_train[i]))
                labels_train.append(labels_train[i])


    # extract features from all data lists
    X_train = extract_features(movements_train)
    X_dev = extract_features(movements_dev)
    X_test = extract_features(movements_test)

    # X is the concatened list of feature vectors, y the label list for X
    return X_train, X_dev, X_test, labels_train, labels_dev, labels_test