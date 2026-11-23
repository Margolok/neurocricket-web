"""Per-trial metrics. Empty completion/latency means censored, never zero."""
from dataclasses import dataclass
import statistics


@dataclass
class TrialResult:
    trial_id: int
    map_id: int
    scenario: str
    map_seed: int
    trial_seed: int
    repetition: int
    algorithm: str
    success: bool
    cancelled: bool
    collision_count: int
    completion_time_s: float | None
    elapsed_simulation_s: float
    path_length_m: float
    average_speed_m_s: float
    spike_count: int
    input_spike_count: int
    response_latency_s: float | None
    danger_events: int
    responded_events: int
    simulation_runtime_s: float
    algorithm_runtime_s: float


def collect(sim, trial_id: int, map_id: int, repetition: int, runtime: float, cancelled: bool = False):
    robot, network = sim.robot, sim.controller.network
    return TrialResult(trial_id, map_id, sim.world.name, sim.world.seed, sim.seed, repetition, sim.algorithm,
                       robot.reached_goal, cancelled, robot.collisions, sim.time if robot.reached_goal else None,
                       sim.time, robot.path_length, robot.path_length / sim.time if sim.time else 0,
                       network.spike_count if network else 0, network.input_spike_count if network else 0,
                       statistics.mean(sim.latencies) if sim.latencies else None, sim.danger_events, len(sim.latencies),
                       runtime, sim.algorithm_runtime)
