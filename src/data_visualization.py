"""visaualize movement data in 3D to inspect the dataset."""

from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt

Coordinate = tuple[int, int, int]
Movement = Sequence[Coordinate]
DEFAULT_AXIS_MARGIN = 1.2


def _compute_axis_limit(
    movements: Sequence[Movement], margin: float = DEFAULT_AXIS_MARGIN
) -> float:
    """Return a symmetric axis limit based on the maximum absolute coordinate."""
    coords = [coord for movement in movements for coord in movement]
    if not coords:
        return 1.0

    max_value = max(
        max(abs(coord[0]), abs(coord[1]), abs(coord[2])) for coord in coords
    )
    return max_value * margin if max_value else 1.0


def visualize_movement(
    movement: Movement,
    title: str = "Movement",
    axis_limit: float | None = None,
    margin: float = DEFAULT_AXIS_MARGIN,
) -> None:
    """Visualize a single movement in 3D."""
    if axis_limit is None:
        axis_limit = _compute_axis_limit([movement], margin=margin)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xs = [coord[0] for coord in movement]
    ys = [coord[1] for coord in movement]
    zs = [coord[2] for coord in movement]

    ax.plot(xs, ys, zs, marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)
    ax.set_box_aspect((1, 1, 1))

    plt.show()


def visualize_movements(
    movements: Iterable[Movement],
    *,
    shared_axis_limit: float | None = None,
    margin: float = DEFAULT_AXIS_MARGIN,
) -> None:
    """Visualize each movement in the iterable with shared axis limits."""
    movements = list(movements)
    if shared_axis_limit is None:
        shared_axis_limit = _compute_axis_limit(movements, margin=margin)

    for index, movement in enumerate(movements, start=1):
        visualize_movement(
            movement, title=f"Movement {index}", axis_limit=shared_axis_limit
        )


if __name__ == "__main__":
    from data_preprocessing import (
        circle_data,
        diagonal_left_data,
        diagonal_right_data,
        horizontal_data,
        vertical_data,
        load_dataset,
    )
    from data_augmentation import augment_movement

    load_dataset()
    data_len = len(circle_data)
    for i in range(data_len):
        circle_data.append(augment_movement(circle_data[i], "circle"))
    visualize_movements(circle_data)
