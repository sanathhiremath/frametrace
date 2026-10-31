"""Owned synthetic pixels and separate truth, excluded from detector input."""
from dataclasses import dataclass
import cv2
import numpy as np
WIDTH, HEIGHT = 320, 180
ZONE = {"x": 170, "y": 35, "width": 110, "height": 110}

@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    truth: bool
    occupied: tuple[int, ...]
    dropped: tuple[int, ...]
    kind: str = "object"

SCENARIOS = (
    Scenario("interrupted-crossing", "An intrusion hidden by dropped frames", True, tuple(range(2, 7)), (3, 5)),
    Scenario("sustained-crossing", "A sustained intrusion both policies catch", True, tuple(range(2, 12)), (3, 5)),
    Scenario("brief-passage", "A brief passage below the event duration", False, (2, 3), ()),
    Scenario("reflected-flashes", "Repeated flashes expose a false alarm", False, (1, 3, 5), (), "flash"),
)

def background() -> np.ndarray:
    pixels = np.full((HEIGHT, WIDTH, 3), (39, 32, 22), dtype=np.uint8)
    for x in range(0, WIDTH, 20):
        cv2.line(pixels, (x, 0), (x, HEIGHT - 1), (48, 40, 30), 1)
    for y in range(0, HEIGHT, 20):
        cv2.line(pixels, (0, y), (WIDTH - 1, y), (48, 40, 30), 1)
    cv2.rectangle(pixels, (170, 35), (280, 145), (72, 95, 80), 1)
    return pixels

def render(scenario: Scenario, index: int) -> np.ndarray:
    if not 0 <= index < 12:
        raise ValueError("frame index must be in [0, 12)")
    pixels = background()
    if index in scenario.occupied:
        color = (160, 240, 210) if scenario.kind == "object" else (230, 230, 230)
        cv2.rectangle(pixels, (194, 65), (220, 112), color, -1)
    else:
        cv2.rectangle(pixels, (30 + index * 4, 65), (56 + index * 4, 112), (160, 240, 210), -1)
    return pixels
