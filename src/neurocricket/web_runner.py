"""Bounded browser trials, using the unchanged Simulation in isolated processes."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from neurocricket.config.settings import Settings
from neurocricket.controllers.base import ALGORITHMS
from neurocricket.simulation.scenarios import SCENARIOS, make_world

MAX_WALL_SECONDS = 90
MAX_MODEL_SECONDS = 60


def validate_request(scenario, algorithm, map_seed, trial_seed, max_time):
    if scenario not in SCENARIOS or algorithm not in ALGORITHMS:
        raise ValueError("Неизвестная карта или алгоритм")
    for value in (map_seed, trial_seed):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 999999:
            raise ValueError("Seed должен быть целым числом от 0 до 999999")
    if isinstance(max_time, bool) or not isinstance(max_time, (int, float)) or not math.isfinite(max_time) or not 0.1 <= max_time <= MAX_MODEL_SECONDS:
        raise ValueError("В браузере доступно не более 60 секунд модельного времени")


def capture_trial(scenario, algorithm, map_seed, trial_seed, max_time=60, wall_seconds=MAX_WALL_SECONDS):
    """Worker-only entry point. Record each existing control tick, not a new model."""
    validate_request(scenario, algorithm, map_seed, trial_seed, max_time)
    from neurocricket.simulation.session import Simulation

    started = time.perf_counter()
    settings = Settings(max_time=float(max_time))
    sim = Simulation(make_world(scenario, map_seed), algorithm, settings, trial_seed)
    frames = []

    def record():
        if not frames or frames[-1]["time_s"] != sim.time:
            frames.append({"time_s": sim.time, "x": sim.robot.x, "y": sim.robot.y,
                "theta": sim.robot.theta, "speed_m_s": sim.robot.speed,
                "distances_m": sim.reading.distances.tolist(),
                "proximity": sim.reading.proximity.tolist(), "endpoints": sim.reading.endpoints,
                "turn": sim.avoidance.turn, "brake": sim.avoidance.brake})

    def cancelled():
        record()
        return time.perf_counter() - started >= wall_seconds

    sim.run(cancelled)
    record()
    timed_out = not sim.done
    from neurocricket.experiments.metrics import collect
    metrics = asdict(collect(sim, 1, 1, 1, time.perf_counter() - started, timed_out))
    network = sim.controller.network
    neural = None
    if network is not None:
        neural = {"spikes": list(network.history.spikes),
            "states": [[t, v.tolist()] for t, v in network.history.states],
            "thresholds": [[t, v.tolist()] for t, v in network.history.thresholds],
            "window_seconds": settings.history_seconds, "sensor_count": settings.sensor_count}
    return {"schema_version": 1, "metrics": metrics, "settings": asdict(settings),
        "world": asdict(sim.world), "frames": frames, "neural": neural,
        "status": "server_timeout" if timed_out else "goal" if sim.robot.reached_goal else "model_limit",
        "backend": "Brian2 numpy CPU / Euler / 1 ms" if network is not None else "Classical controller"}


def run_isolated(scenario, algorithm, map_seed, trial_seed, max_time=60):
    """No shared Brian2 globals across sessions; enforce a hard parent timeout."""
    validate_request(scenario, algorithm, map_seed, trial_seed, max_time)
    request = {"scenario": scenario, "algorithm": algorithm, "map_seed": map_seed,
        "trial_seed": trial_seed, "max_time": max_time}
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    try:
        completed = subprocess.run([sys.executable, "-m", "neurocricket.web_runner"],
            input=json.dumps(request), capture_output=True, text=True, encoding="utf-8",
            env=env, timeout=MAX_WALL_SECONDS + 10, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Сервер остановил слишком долгий расчёт. Попробуйте меньший лимит времени.") from exc
    if completed.returncode:
        # Keep technical diagnostics on the server, not in the public interface.
        print(completed.stderr[-4000:], file=sys.stderr)
        raise RuntimeError("Не удалось выполнить расчёт. Попробуйте ещё раз или выберите другую карту.")
    try:
        result = json.loads(completed.stdout)
        if result["schema_version"] != 1 or not result["frames"]:
            raise ValueError("Invalid worker result")
        return result
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Сервер вернул неполный результат расчёта.") from exc


if __name__ == "__main__":
    # Only a validated JSON request enters the worker, never arbitrary code or paths.
    print(json.dumps(capture_trial(**json.load(sys.stdin)), ensure_ascii=False, allow_nan=False))
