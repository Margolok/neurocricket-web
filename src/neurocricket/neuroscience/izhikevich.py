"""Separate intrinsic-firing demonstration; not used to claim LIF bursts."""
from pathlib import Path
import numpy as np
import brian2 as b2


def simulate_demo():
    """Original dimensionless Izhikevich variables, time expressed in ms."""
    b2.prefs.codegen.target = "numpy"
    clock = b2.Clock(dt=0.25 * b2.ms)
    cells = b2.NeuronGroup(2, """
        dv/dt = (0.04*v**2 + 5*v + 140 - u + stimulus)/ms : 1
        du/dt = a*(b*v-u)/ms : 1
        a : 1
        b : 1
        c : 1
        d : 1
        stimulus : 1
        """, threshold="v >= 30", reset="v = c; u += d", method="euler", clock=clock)
    cells.a, cells.b = 0.02, 0.2
    cells.c, cells.d = [-65, -50], [8, 2]  # regular spiking and chattering
    cells.v = -65
    cells.u = "b*v"
    cells.stimulus = 10
    states, spikes = b2.StateMonitor(cells, "v", record=True), b2.SpikeMonitor(cells)
    net = b2.Network(cells, states, spikes)
    net.run(300 * b2.ms, namespace={})
    return np.asarray(states.t[:] / b2.ms), np.asarray(states.v[:]), {
        i: np.asarray(spikes.spike_trains()[i] / b2.ms) for i in range(2)}


def export_demo(root: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from neurocricket.experiments.exporter import write_json
    root.mkdir(parents=True, exist_ok=True)
    times, voltages, spikes = simulate_demo()
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), layout="constrained")
        for i, ax in enumerate(axes):
            ax.plot(times, voltages[i], color=["#50d8ef", "#bca0ff"][i], lw=1)
            ax.set(title=["Izhikevich: regular spiking", "Izhikevich: chattering / intrinsic bursts"][i], xlabel="t, мс", ylabel="v (условная шкала)")
        output = root / "izhikevich_demo.png"
        fig.savefig(output, dpi=140)
        plt.close(fig)
    write_json(root / "izhikevich_demo.json", {"spikes_ms": {str(k): v.tolist() for k, v in spikes.items()},
                                              "parameters": {"a": .02, "b": .2, "c": [-65, -50], "d": [8, 2], "I": 10, "dt_ms": .25},
                                              "note": "Standalone illustrative model, not robot controller or exact cricket physiology"})
    return output
