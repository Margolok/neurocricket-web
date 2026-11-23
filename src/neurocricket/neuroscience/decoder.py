from dataclasses import dataclass
import numpy as np


@dataclass
class Avoidance:
    turn: float = 0.0
    brake: float = 0.0
    forward: float = 1.0
    intensity: float = 0.0


class MotorDecoder:
    """Exponential motor-rate readout; no distance-based control shortcut."""
    def __init__(self):
        self.rates = np.zeros(4)

    def decode(self, counts: np.ndarray, dt: float) -> Avoidance:
        alpha = 1 - np.exp(-dt / 0.15)
        self.rates = (1 - alpha) * self.rates + alpha * counts / dt
        left, right, forward, brake = self.rates
        turn = float(np.clip((right - left) / 35, -1, 1))
        danger = float(np.clip((left + right) / 70 + brake / 100, 0, 1))
        return Avoidance(turn, float(np.clip(brake / 55, 0, 1)), float(np.clip(forward / 30, 0, 1)), danger)
