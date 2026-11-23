"""Matched-budget temporal encoding; deterministic residual integration, no noise."""
from dataclasses import dataclass
import numpy as np
from neurocricket.config.settings import NeuralSettings


@dataclass
class SpikeSchedule:
    counts: np.ndarray
    base: np.ndarray
    interval: np.ndarray
    within: np.ndarray
    size: np.ndarray

    def times(self, sensor: int) -> np.ndarray:
        k = np.arange(self.counts[sensor])
        return self.base[sensor] + (k // self.size[sensor]) * self.interval[sensor] + (k % self.size[sensor]) * self.within[sensor]


class TemporalEncoder:
    """Both codes spend floor(integrated rate) spikes, per sensor and window.

    At critical proximity Burst packs this same budget into groups of up to 3.
    Input sources are event encoders, not claimed to be biological LIF cells.
    """
    def __init__(self, mode: str, count: int, settings: NeuralSettings, seed: int):
        settings.validate()
        if mode not in ("rate", "burst"):
            raise ValueError(mode)
        self.mode, self.settings = mode, settings
        self.residual = np.random.default_rng(seed).uniform(0, 1, count)

    def schedule(self, proximity: np.ndarray, time: float, dt: float) -> SpikeSchedule:
        values = np.asarray(proximity, dtype=float)
        if values.shape != self.residual.shape or not np.all(np.isfinite(values)):
            raise ValueError("Ожидается конечное показание для каждого сенсора")
        if not np.isfinite(time) or time < 0 or dt != 0.1:
            raise ValueError("Время должно быть неотрицательным; окно кодирования — 100 мс")
        p = np.clip(values, 0, 1)
        rates = self.settings.f_min_hz + p * (self.settings.f_max_hz - self.settings.f_min_hz)
        self.residual += rates * dt
        counts = np.floor(self.residual + 1e-10).astype(int)
        self.residual -= counts
        size = np.ones(len(p), dtype=int)
        if self.mode == "burst":
            size[p >= self.settings.burst_threshold] = self.settings.burst_size
        clusters = np.maximum(1, np.ceil(counts / size))
        interval = (dt - 0.002) / clusters
        within = np.where(size > 1, self.settings.burst_isi_ms / 1000, 0)
        return SpikeSchedule(counts, np.full(len(p), time + 0.001), interval, within, size)
