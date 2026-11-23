"""Validated, serializable physical and neural parameters (SI unless specified)."""
from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Integral, Real
from pathlib import Path


def validate_numbers(values: dict, integer_fields=()) -> None:
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"Параметр {key} должен быть конечным числом")
        if key in integer_fields and not isinstance(value, Integral):
            raise ValueError(f"Параметр {key} должен быть целым числом")


@dataclass
class NeuralSettings:
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 12.0
    rest_mv: float = -65.0
    threshold_mv: float = -50.0
    reset_mv: float = -65.0
    refractory_ms: float = 2.0
    input_weight_mv: float = 36.0
    motor_weight_mv: float = 40.0
    inhibition_mv: float = 10.0
    f_min_hz: float = 0.0
    f_max_hz: float = 100.0
    burst_threshold: float = 0.65
    burst_size: int = 3
    burst_isi_ms: float = 2.0
    adaptive_increment_mv: float = 0.0
    adaptive_tau_ms: float = 200.0

    def validate(self) -> None:
        validate_numbers(asdict(self), ("burst_size",))
        if not (1 <= self.tau_m_ms <= 100 and 1 <= self.tau_syn_ms <= 100):
            raise ValueError("Постоянные времени: 1–100 мс")
        if not (-100 <= self.rest_mv < self.threshold_mv <= 0 and -100 <= self.reset_mv < self.threshold_mv):
            raise ValueError("Неверные потенциалы LIF")
        if not (1 <= self.refractory_ms <= 20 and 0 <= self.f_min_hz <= self.f_max_hz <= 100):
            raise ValueError("Неверная частота или рефрактерность")
        if not (0 < self.burst_threshold <= 1 and 2 <= self.burst_size <= 5 and 1 <= self.burst_isi_ms <= 3):
            raise ValueError("Неверные параметры пачки")
        if not all(0 <= v <= 100 for v in (self.input_weight_mv, self.motor_weight_mv, self.inhibition_mv)):
            raise ValueError("Веса: 0–100 мВ")
        if not (0 <= self.adaptive_increment_mv <= 5 and 10 <= self.adaptive_tau_ms <= 1000):
            raise ValueError("Неверная адаптация порога")


@dataclass
class Settings:
    control_dt: float = 0.1
    neural_dt: float = 0.001
    robot_radius: float = 0.12
    robot_speed: float = 0.8
    max_angular_speed: float = 2.8
    sensor_range: float = 2.0
    sensor_count: int = 5
    goal_radius: float = 0.25
    max_time: float = 60.0
    danger_threshold: float = 0.65
    response_turn_threshold: float = 0.3
    history_seconds: float = 5.0
    neural: NeuralSettings = field(default_factory=NeuralSettings)

    def validate(self) -> None:
        if not isinstance(self.neural, NeuralSettings):
            raise ValueError("neural должен содержать нейронную конфигурацию")
        validate_numbers({key: value for key, value in asdict(self).items() if key != "neural"}, ("sensor_count",))
        if self.control_dt != 0.1 or self.neural_dt != 0.001:
            raise ValueError("Детерминированный протокол: управление 100 мс, Brian2 1 мс")
        if not (5 <= self.sensor_count <= 9 and self.sensor_count % 2 == 1):
            raise ValueError("Число датчиков: 5, 7 или 9")
        if not (0.05 <= self.robot_radius <= 0.25 and 0.1 <= self.robot_speed <= 2):
            raise ValueError("Неверный радиус или скорость робота")
        if not (0.5 <= self.sensor_range <= 4 and 0.05 <= self.goal_radius <= 0.5):
            raise ValueError("Неверная дальность или радиус цели")
        if not (0.1 <= self.max_angular_speed <= 6 and 0.1 <= self.max_time <= 600):
            raise ValueError("Неверное ограничение скорости/времени")
        if not (0 < self.danger_threshold <= 1 and 0 < self.response_turn_threshold <= 6 and 1 <= self.history_seconds <= 20):
            raise ValueError("Неверные пороги или длина истории")
        self.neural.validate()

    def save(self, path: Path) -> None:
        self.validate()
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        if not isinstance(data, dict) or not isinstance(data.get("neural", {}), dict):
            raise ValueError("Конфигурация и neural должны быть JSON-объектами")
        try:
            settings = cls(**{**data, "neural": NeuralSettings(**data.get("neural", {}))})
        except TypeError as exc:
            raise ValueError(f"Неизвестный параметр конфигурации: {exc}") from exc
        settings.validate()
        return settings

    @classmethod
    def load(cls, path: Path) -> "Settings":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))
