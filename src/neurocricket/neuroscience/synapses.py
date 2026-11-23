"""Explicit signed connectivity; indices in the common LIF group."""
import numpy as np
from neurocricket.config.settings import NeuralSettings


def topology(angles: np.ndarray, cfg: NeuralSettings):
    n = len(angles)
    left, right, forward, brake = range(2 * n, 2 * n + 4)
    edges = []
    for i, angle in enumerate(angles):
        # Sensor relay -> inhibitory interneuron: feedforward opponent inhibition.
        edges.append((i, n + i, cfg.motor_weight_mv))
        if abs(angle) < 1e-8:
            edges.extend([(i, brake, cfg.motor_weight_mv * 1.3), (i, right, cfg.motor_weight_mv),
                          (i, left, cfg.motor_weight_mv * 0.45), (n + i, forward, -cfg.inhibition_mv)])
        else:
            away, toward = (right, left) if angle < 0 else (left, right)
            gain = 1.0 if abs(angle) < 1.0 else 0.45
            edges.extend([(i, away, cfg.motor_weight_mv * gain), (n + i, toward, -cfg.inhibition_mv * gain)])
    edges.extend([(left, right, -cfg.inhibition_mv * 0.3), (right, left, -cfg.inhibition_mv * 0.3),
                  (brake, forward, -cfg.inhibition_mv * 1.5)])
    return edges
