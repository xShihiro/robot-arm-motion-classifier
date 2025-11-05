import os

# extract the XYZ coordinates and create a list of ordered tuples
def movement_into_tuple_list(filepath: str) -> list:
    res = []
    with open(filepath, "r") as file:
        content = file.read().replace("\n", "")
        lines = content.split("#")
        
    for line in lines:
        parts = line.split(",")
        if len(parts) > 6:
            values = parts[6].split("/")
            res.append(tuple(int(v) for v in values))
    
    return res