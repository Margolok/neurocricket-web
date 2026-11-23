from dataclasses import dataclass
import math
import numpy as np
from .geometry import ray_rect
from .robot import Robot
from .world import World


@dataclass
class SensorReading:
    distances: np.ndarray
    proximity: np.ndarray
    angles: np.ndarray
    endpoints: list[tuple[float, float]]


def sensor_angles(count: int = 5) -> np.ndarray:
    if count < 5 or count % 2 == 0:
        raise ValueError("Требуется нечётное число датчиков >=5")
    # Order: front-left, centre, front-right, side-left, side-right, extras.
    base = [-math.pi / 4, 0, math.pi / 4, -math.pi / 2, math.pi / 2]
    for i in range((count - 5) // 2):
        a = math.pi / (6 + 2 * i)
        base.extend([-a, a])
    return np.array(base)


def sense(robot: Robot, world: World, max_distance: float, count: int = 5) -> SensorReading:
    angles = sensor_angles(count)
    distances, endpoints = [], []
    for a in angles:
        dx, dy = math.cos(robot.theta + a), math.sin(robot.theta + a)
        # Range and reported clearance start at the surface of the robot.
        origin = robot.x + robot.radius * dx, robot.y + robot.radius * dy
        d = min(max_distance, *(ray_rect(origin, (dx, dy), r) for r in world.obstacles + world.walls))
        distances.append(d)
        endpoints.append((origin[0] + d * dx, origin[1] + d * dy))
    ds = np.array(distances)
    return SensorReading(ds, np.clip(1 - ds / max_distance, 0, 1), angles, endpoints)
