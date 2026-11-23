"""True Brian2 LIF network with physical units and signed current synapses."""
import numpy as np
import brian2 as b2
from neurocricket.config.settings import Settings
from neurocricket.simulation.sensors import sensor_angles
from .neurons import LIF_EQUATIONS, INPUT_EQUATIONS, INPUT_THRESHOLD
from .synapses import topology
from .encoders import TemporalEncoder
from .monitors import NeuralHistory
from .decoder import MotorDecoder

b2.prefs.codegen.target = "numpy"  # CPU, no external compiler required.


class SNNNetwork:
    def __init__(self, mode: str, settings: Settings, seed: int = 42):
        settings.validate()
        self.settings, self.n, self.mode = settings, settings.sensor_count, mode
        c = settings.neural
        self.clock = b2.Clock(dt=settings.neural_dt * b2.second)
        namespace = {"rest": c.rest_mv * b2.mV, "threshold": c.threshold_mv * b2.mV,
                     "reset_v": c.reset_mv * b2.mV, "tau_m": c.tau_m_ms * b2.ms,
                     "tau_syn": c.tau_syn_ms * b2.ms, "tau_adapt": c.adaptive_tau_ms * b2.ms,
                     "increment": c.adaptive_increment_mv * b2.mV}
        self.inputs = b2.NeuronGroup(self.n, INPUT_EQUATIONS, threshold=INPUT_THRESHOLD,
                                     reset="cursor += 1", clock=self.clock)
        self.inputs.size = 1
        self.cells = b2.NeuronGroup(2 * self.n + 4, LIF_EQUATIONS, threshold="v >= threshold + adapt",
                                    reset="v = reset_v; adapt += increment", refractory=c.refractory_ms * b2.ms,
                                    method="euler", namespace=namespace, clock=self.clock)
        self.cells.v = c.rest_mv * b2.mV
        self.cells.drive[2 * self.n + 2] = 21 * b2.mV
        self.input_synapses = b2.Synapses(self.inputs, self.cells, "w : volt", on_pre="syn_post += w", clock=self.clock)
        self.input_synapses.connect(i=np.arange(self.n), j=np.arange(self.n))
        self.input_synapses.w = c.input_weight_mv * b2.mV
        self.edges = topology(sensor_angles(self.n), c)
        self.synapses = b2.Synapses(self.cells, self.cells, "w : volt", on_pre="syn_post += w", clock=self.clock)
        self.synapses.connect(i=[e[0] for e in self.edges], j=[e[1] for e in self.edges])
        self.synapses.w = np.array([e[2] for e in self.edges]) * b2.mV
        self.spikes = b2.SpikeMonitor(self.cells)
        self.input_spikes = b2.SpikeMonitor(self.inputs)
        self.states = b2.StateMonitor(self.cells, ["v", "adapt", "syn"], record=True, dt=5 * b2.ms)
        self.net = b2.Network(self.inputs, self.cells, self.input_synapses, self.synapses, self.spikes, self.input_spikes, self.states)
        self.encoder = TemporalEncoder(mode, self.n, c, seed)
        self.decoder = MotorDecoder()
        self.history = NeuralHistory(2 * self.n + 4, settings.history_seconds)
        self.last_proximity = np.zeros(self.n)
        self.last_budget = np.zeros(self.n, dtype=int)
        self.last_counts = np.zeros(2 * self.n + 4, dtype=int)

    @property
    def spike_count(self) -> int:
        return int(np.sum(self.spikes.count[:]) + np.sum(self.input_spikes.count[:]))

    @property
    def input_spike_count(self) -> int:
        return int(np.sum(self.input_spikes.count[:]))

    def prepare(self, proximity: np.ndarray) -> None:
        s = self.encoder.schedule(proximity, float(self.net.t / b2.second), self.settings.control_dt)
        self.last_proximity = np.clip(np.asarray(proximity, dtype=float), 0, 1).copy()
        self.last_budget = s.counts.copy()
        self.inputs.cursor = 0
        self.inputs.total = s.counts
        self.inputs.base = s.base * b2.second
        self.inputs.interval = s.interval * b2.second
        self.inputs.within = s.within * b2.second
        self.inputs.size = s.size

    def readout(self):
        counts = self.history.collect(self)
        self.last_counts = counts.copy()
        return self.decoder.decode(counts[-4:], self.settings.control_dt)

    def connections(self):
        """Actual current synaptic weights in the common input/E/I/motor IDs."""
        edges = [(int(i), int(j) + self.n, float(w)) for i, j, w in zip(
            self.input_synapses.i[:], self.input_synapses.j[:], self.input_synapses.w[:] / b2.mV)]
        edges.extend((int(i) + self.n, int(j) + self.n, float(w)) for i, j, w in zip(
            self.synapses.i[:], self.synapses.j[:], self.synapses.w[:] / b2.mV))
        return edges

    def snapshot(self, selected):
        if not 0 <= selected < 3 * self.n + 4:
            raise ValueError("Нейрон вне диапазона сети")
        counts = np.concatenate((np.asarray(self.input_spikes.count[:], dtype=int), np.asarray(self.spikes.count[:], dtype=int)))
        recent = [t for t, i in self.history.spikes if i == selected]
        intervals = np.diff(recent) * 1000
        data = {"neuron": selected, "spike_count": int(counts[selected]),
                "window_spikes": len(recent), "window_duration_s": min(float(self.net.t / b2.second), self.settings.history_seconds),
                "median_isi_ms": float(np.median(intervals)) if len(intervals) else None,
                "motor_rates_hz": self.decoder.rates.tolist(),
                "layer_counts": [int(sum(counts[lo:hi])) for lo, hi in ((0, self.n), (self.n, 2*self.n), (2*self.n, 3*self.n), (3*self.n, 3*self.n+4))]}
        edges = self.connections()
        data["incoming"] = [(source, weight) for source, target, weight in edges if target == selected]
        data["outgoing"] = [(target, weight) for source, target, weight in edges if source == selected]
        if selected < self.n:
            data.update({"proximity": float(self.last_proximity[selected]), "budget": int(self.last_budget[selected])})
        else:
            cell = selected - self.n
            data.update({"voltage_mv": float(self.cells.v[cell] / b2.mV),
                "threshold_mv": self.settings.neural.threshold_mv + float(self.cells.adapt[cell] / b2.mV),
                "synaptic_mv": float(self.cells.syn[cell] / b2.mV)})
        return data

    def advance(self, proximity: np.ndarray):
        self.prepare(proximity)
        self.net.run(self.settings.control_dt * b2.second, namespace={})
        return self.readout()
