"""Shared goal attraction and actuator mapping, identical in all three modes."""
import math
from neurocricket.config.settings import Settings
from neurocricket.simulation.geometry import wrap_angle
from neurocricket.neuroscience.decoder import Avoidance

ALGORITHMS = ("classical", "rate", "burst")
ALGORITHM_LABELS = {"classical": "Classical", "rate": "SNN Rate", "burst": "SNN Burst"}


def motor_command(robot, goal, avoidance: Avoidance, cfg: Settings) -> tuple[float, float]:
    error = wrap_angle(math.atan2(goal[1] - robot.y, goal[0] - robot.x) - robot.theta)
    attraction = max(-cfg.max_angular_speed, min(cfg.max_angular_speed, 2.2 * error))
    omega = attraction * (1 - avoidance.intensity) + cfg.max_angular_speed * avoidance.turn
    omega = max(-cfg.max_angular_speed, min(cfg.max_angular_speed, omega))
    speed = cfg.robot_speed * avoidance.forward * (1 - 0.95 * avoidance.brake)
    speed *= max(0.12, 1 - 0.7 * abs(omega) / cfg.max_angular_speed)
    if avoidance.brake >= 0.95:
        speed = 0.0
    return speed, omega


def make_controller(algorithm: str, cfg: Settings, seed: int):
    if algorithm == "classical":
        from .classical import ClassicalController
        return ClassicalController(cfg)
    if algorithm in ("rate", "burst"):
        from .snn_rate import SNNController
        return SNNController(algorithm, cfg, seed)
    raise ValueError(f"Неизвестный алгоритм: {algorithm}")
