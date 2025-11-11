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

# exctract all features from data, initialize one X and create a corresponding label list y
def prepare_all_data():
    # extract features from all data lists
    circle_features = extract_features(circle_data)
    diagonal_left_features = extract_features(diagonal_left_data)
    diagonal_right_features = extract_features(diagonal_right_data)
    horizontal_features = extract_features(horizontal_data)
    vertical_features = extract_features(vertical_data)
    
    # combine the data
    X = (
         circle_features +
         diagonal_left_features + 
         diagonal_right_features +
         horizontal_features +
         vertical_features
    )
    
    # create the label list
    y = (
        ["circle"] * len(circle_features) +
        ["diagonal_left"] * len(diagonal_left_features) +
        ["diagonal_right"] * len(diagonal_right_features) +
        ["horizontal"] * len(horizontal_features) +
        ["vertical"] * len(vertical_features)
    )
    
    # X is the concatened list of feature vectors, y the label list for X
    return X, y