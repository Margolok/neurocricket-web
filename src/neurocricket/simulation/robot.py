from dataclasses import dataclass, field
from collections import deque


@dataclass
class Robot:
    x: float
    y: float
    theta: float = 0.0
    radius: float = 0.12
    speed: float = 0.0
    angular_speed: float = 0.0
    collided: bool = False
    reached_goal: bool = False
    collisions: int = 0
    path_length: float = 0.0
    trajectory: deque = field(default_factory=lambda: deque(maxlen=20000))

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y
