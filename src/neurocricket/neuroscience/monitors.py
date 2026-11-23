"""Bounded recording adapter for the pinned Brian2 2.9 runtime backend."""
from collections import deque
import numpy as np
from brian2 import mV, second


class NeuralHistory:
    def __init__(self, count: int, seconds: float):
        self.seconds = seconds
        self.spikes = deque(maxlen=50000)
        self.states = deque(maxlen=int(seconds / 0.005) + 30)
        self.thresholds = deque(maxlen=self.states.maxlen)
        self.synaptic = deque(maxlen=self.states.maxlen)
        self.last_counts = np.zeros(count, dtype=int)
        self.last_input_counts = np.zeros((count - 4) // 2, dtype=int)
        self.input_delta = np.zeros_like(self.last_input_counts)

    def collect(self, network) -> np.ndarray:
        counts = np.asarray(network.spikes.count[:], dtype=int)
        delta = counts - self.last_counts
        self.last_counts = counts.copy()
        inputs = np.asarray(network.input_spikes.count[:], dtype=int)
        self.input_delta = inputs - self.last_input_counts
        self.last_input_counts = inputs.copy()
        events = [(float(t), int(i)) for i, t in zip(network.input_spikes.i[:], network.input_spikes.t[:] / second)]
        events.extend((float(t), int(i) + network.n) for i, t in zip(network.spikes.i[:], network.spikes.t[:] / second))
        self.spikes.extend(sorted(events))
        state = np.asarray(network.states.v[:] / mV)
        thresholds = network.settings.neural.threshold_mv + np.asarray(network.states.adapt[:] / mV)
        synaptic = np.asarray(network.states.syn[:] / mV)
        for col, t in enumerate(network.states.t[:] / second):
            self.states.append((float(t), state[:, col].copy()))
            self.thresholds.append((float(t), thresholds[:, col].copy()))
            self.synaptic.append((float(t), synaptic[:, col].copy()))
        now = float(network.net.t / second)
        while self.spikes and self.spikes[0][0] < now - self.seconds:
            self.spikes.popleft()
        for history in (self.states, self.thresholds, self.synaptic):
            while history and history[0][0] < now - self.seconds:
                history.popleft()
        # Brian2 SpikeMonitor.reinit is unimplemented. resize does not reset N.
        # Clear recording arrays only; per-cell cumulative count is preserved.
        for monitor in (network.spikes, network.input_spikes):
            monitor.resize(0)
            monitor.variables["N"].set_value(0)
        network.states.resize(0)
        return delta
