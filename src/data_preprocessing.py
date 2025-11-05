import os

# lists to fill the corresponding data in
circle_data = []
diagonal_left_data = []
diagonal_right_data = []
horizontal = []
vertical = []

# extract the XYZ coordinates and create a list of ordered tuples
def movement_into_tuple_list(filepath: str) -> list:
    res = []
    
    # create a list of all the lines in the file
    with open(filepath, "r") as file:
        content = file.read().replace("\n", "")
        lines = content.split("#")
    
    # for each line extract the column with the coordinates, converts it into
    # an int tuple and appends it to the result list    
    for line in lines:
        parts = line.split(",")
        if len(parts) > 6:
            values = parts[6].split("/")
            res.append(tuple(int(v) for v in values))
    
    return res