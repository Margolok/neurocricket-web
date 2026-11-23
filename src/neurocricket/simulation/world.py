from dataclasses import dataclass
from .geometry import Rect


@dataclass(frozen=True)
class World:
    name: str
    seed: int
    obstacles: tuple[Rect, ...]
    start: tuple[float, float] = (0.7, 3.0)
    goal: tuple[float, float] = (8.6, 3.0)
    width: float = 10.0
    height: float = 6.0

    @property
    def walls(self) -> tuple[Rect, ...]:
        return (Rect(-1, -1, self.width + 2, 1), Rect(-1, self.height, self.width + 2, 1),
                Rect(-1, 0, 1, self.height), Rect(self.width, 0, 1, self.height))

    def collides(self, x: float, y: float, radius: float) -> bool:
        return (x - radius <= 0 or x + radius >= self.width or y - radius <= 0 or y + radius >= self.height
                or any(r.circle_intersects(x, y, radius) for r in self.obstacles))
