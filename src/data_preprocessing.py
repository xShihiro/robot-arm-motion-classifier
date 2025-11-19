from pathlib import Path
from typing import Dict, List, Tuple

Movement = List[Tuple[int, int, int]]

# important: use the path of the data directory you want to use as Path parameter
project_root = Path(__file__).parent.parent
data_directory = project_root / "dataset_combined"

# lists to fill the corresponding data in
circle_data: List[Movement] = []
diagonal_left_data: List[Movement] = []
diagonal_right_data: List[Movement] = []
horizontal_data: List[Movement] = []
vertical_data: List[Movement] = []

# Map directory names to list names
DIR_TO_LIST: Dict[str, List[Movement]] = {
    "circle": circle_data,
    "diagonal_left": diagonal_left_data,
    "diagonal_right": diagonal_right_data,
    "horizontal": horizontal_data,
    "vertical": vertical_data,
}


def _movement_into_tuple_list(filepath: str) -> Movement:
    """Extract the XYZ coordinates from a single file into a list of tuples."""
    res: Movement = []

    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read().replace("\n", "")
        lines = content.split("#")

    for line in lines:
        parts = line.split(",")
        if len(parts) > 6:
            values = parts[6].split("/")
            res.append(tuple(int(v) for v in values))

    return res


def _fill_data_lists(dataset_path: Path):
    """Extract all coordinates from the dataset and append them to the class lists."""
    for directory in sorted(dataset_path.iterdir()):
        target_list = DIR_TO_LIST.get(directory.name)

        for filename in sorted(directory.iterdir()):
            target_list.append(_movement_into_tuple_list(str(filename)))


def load_dataset(dataset_path: Path = data_directory):
    """Reload the movement datasets from disk."""
    for data_list in DIR_TO_LIST.values():
        data_list.clear()
    _fill_data_lists(dataset_path)
