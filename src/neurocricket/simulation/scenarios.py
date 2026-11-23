"""Four scenario families. Map seed is independent from trial/neural seed."""
import numpy as np
from .geometry import Rect
from .world import World

SCENARIOS = ("open", "corridor", "maze", "random")
SCENARIO_LABELS = {"open": "Открытое пространство", "corridor": "Узкий коридор", "maze": "Простой лабиринт", "random": "Случайные препятствия"}


def make_world(name: str, seed: int = 42) -> World:
    if name not in SCENARIOS:
        raise ValueError(f"Неизвестная карта: {name}")
    rng = np.random.default_rng(seed)
    if name == "open":
        obstacles = []
    elif name == "corridor":
        shift = float(rng.uniform(-0.15, 0.15))
        obstacles = [Rect(1.5, 0, 7, 2.3 + shift), Rect(1.5, 3.7 + shift, 7, 2.3 - shift), Rect(4.8, 2.3 + shift, 0.35, 0.45)]
    elif name == "maze":
        shift = float(rng.uniform(-0.2, 0.2))
        obstacles = [Rect(3, 0, 0.25, 3.5 + shift), Rect(6, 2.5 + shift, 0.25, 3.5 - shift), Rect(7.7, 0.8, 0.5, 1)]
    else:
        obstacles = []
        for _ in range(250):
            if len(obstacles) == 12:
                break
            r = Rect(float(rng.uniform(1.6, 8.3)), float(rng.uniform(0.4, 5)), float(rng.uniform(0.3, 0.65)), float(rng.uniform(0.3, 0.7)))
            if r.circle_intersects(8.6, 3.0, 0.5):
                continue
            if any(Rect(o.x - 0.3, o.y - 0.3, o.w + 0.6, o.h + 0.6).circle_intersects(r.x + r.w / 2, r.y + r.h / 2, max(r.w, r.h) / 2) for o in obstacles):
                continue
            obstacles.append(r)
    if name == "open":
        y = float(rng.uniform(2.3, 3.7))
        return World(name, seed, tuple(obstacles), start=(0.7, y), goal=(8.6, y))
    return World(name, seed, tuple(obstacles))


def map_suite(count: int, seed: int) -> list[World]:
    if not 1 <= count <= 1000:
        raise ValueError("Количество карт: 1–1000")
    return [make_world(SCENARIOS[i % 4], seed + i) for i in range(count)]
