from pathlib import Path

# important: use the path of the data directory you wanna useas Path parameter
data_directory = Path("dataset#1")

# lists to fill the corresponding data in
circle_data = []
diagonal_left_data = []
diagonal_right_data = []
horizontal_data = []
vertical_data = []

# Map directory names to list names
dir_to_list = {
    "circle": circle_data,
    "diagonal_left": diagonal_left_data, 
    "diagonal_right": diagonal_right_data,
    "horizontal": horizontal_data,
    "vertical": vertical_data
}

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

# extract all the coordinates from a dataset and append to the rights data lists
def fill_data_lists(dir: str):
    
    # iterate over every data class and get the corresponding list name
    for directory in sorted(dir.iterdir()):
        target_list = dir_to_list[directory.name]
        
        # iterate over every file and append a movement sublist to the data list
        for filename in sorted(directory.iterdir()):
            target_list.append(movement_into_tuple_list(str(filename)))
            

# fill the data lists with the chosen dataset
fill_data_lists(data_directory)