import numpy as np
from neurocricket.neuroscience.decoder import Avoidance


class ClassicalController:
    network = None

    def __init__(self, settings):
        self.settings = settings

    def step(self, reading) -> Avoidance:
        p = reading.proximity
        directional = np.sin(reading.angles)
        side = float(-np.dot(p ** 3, directional))
        centre = float(p[1])
        # Fixed right bias breaks symmetric head-on encounters.
        turn = np.clip(2.5 * side + 0.9 * centre ** 4, -1, 1)
        danger = float(max(p[:3]) ** 2)
        brake = float(np.clip((centre - 0.45) / 0.5, 0, 1))
        return Avoidance(float(turn), brake, 1.0, danger)
