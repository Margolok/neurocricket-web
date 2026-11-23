"""Analytical ray/AABB intersections and exact circle/AABB contacts."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    def circle_intersects(self, x: float, y: float, radius: float) -> bool:
        qx = min(max(x, self.x), self.x + self.w)
        qy = min(max(y, self.y), self.y + self.h)
        return (x - qx) ** 2 + (y - qy) ** 2 <= radius ** 2


def ray_rect(origin: tuple[float, float], direction: tuple[float, float], rect: Rect) -> float:
    """Distance along a unit ray; infinity for a miss, zero if inside."""
    near, far = 0.0, math.inf
    for p, d, low, high in zip(origin, direction, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h)):
        if abs(d) < 1e-12:
            if p < low or p > high:
                return math.inf
        else:
            a, b = (low - p) / d, (high - p) / d
            near, far = max(near, min(a, b)), min(far, max(a, b))
            if near > far:
                return math.inf
    return near


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
