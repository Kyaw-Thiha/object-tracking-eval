from __future__ import annotations

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
