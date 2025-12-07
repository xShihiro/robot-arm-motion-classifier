from pathlib import Path

Movement = list[tuple[int, int, int]]

# important: use the path of the data directory you want to use as Path parameter
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "dataset_combined"

# lists to fill the corresponding data in
circle_data: list[Movement] = []
diagonal_left_data: list[Movement] = []
diagonal_right_data: list[Movement] = []
horizontal_data: list[Movement] = []
vertical_data: list[Movement] = []

# Map directory names to list names
DIR_TO_LIST: dict[str, list[Movement]] = {
    "circle": circle_data,
    "diagonal_left": diagonal_left_data,
    "diagonal_right": diagonal_right_data,
    "horizontal": horizontal_data,
    "vertical": vertical_data,
}


def movement_into_tuple_list(filepath: str) -> Movement:
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


def _fill_data_lists(dataset_path: Path) -> None:
    """Extract all coordinates from the dataset and append them to the class lists."""
    for directory in sorted(dataset_path.iterdir()):
        target_list = DIR_TO_LIST.get(directory.name)

        for filename in sorted(directory.iterdir()):
            target_list.append(movement_into_tuple_list(str(filename)))


def load_dataset(dataset_path: Path = DATA_DIRECTORY) -> None:
    """Reload the movement datasets from disk."""
    for data_list in DIR_TO_LIST.values():
        data_list.clear()
    _fill_data_lists(dataset_path)
