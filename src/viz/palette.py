from __future__ import annotations

from colorsys import hsv_to_rgb

CLASS_COLORS: dict[int, tuple[float, float, float]] = {
    0: (0.90, 0.10, 0.10),  # car
    1: (0.90, 0.45, 0.10),  # truck
    2: (0.95, 0.75, 0.15),  # bus
    3: (0.55, 0.35, 0.10),  # trailer
    4: (0.65, 0.65, 0.20),  # construction
    5: (0.10, 0.75, 0.25),  # pedestrian
    6: (0.10, 0.65, 0.75),  # motorcycle
    7: (0.15, 0.35, 0.95),  # bicycle
    8: (0.75, 0.15, 0.85),  # trafficcone
    9: (0.55, 0.55, 0.55),  # barrier
}
DEFAULT_COLOR: tuple[float, float, float] = (0.8, 0.8, 0.8)


def generate_palette(
    size: int,
    start_hue: float = 0.0,
    saturation: float = 0.65,
    value: float = 0.95,
) -> dict[int, tuple[float, float, float]]:
    palette: dict[int, tuple[float, float, float]] = {}
    golden_ratio = 0.61803398875
    hue = start_hue
    for idx in range(size):
        hue = (hue + golden_ratio) % 1.0
        r, g, b = hsv_to_rgb(hue, saturation, value)
        palette[idx] = (r, g, b)
    return palette


COCO_CLASS_COLORS = generate_palette(80)


def palette_for_dataset(dataset_name: str | None) -> dict[int, tuple[float, float, float]]:
    if dataset_name in {"camel", "coco"}:
        return COCO_CLASS_COLORS
    return CLASS_COLORS
