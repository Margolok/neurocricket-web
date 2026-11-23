"""One simulation session, shared by GUI and headless experiments."""
import time
from copy import deepcopy
from neurocricket.config.settings import Settings
from neurocricket.controllers.base import make_controller, motor_command
from neurocricket.neuroscience.decoder import Avoidance
from .robot import Robot
from .sensors import sense
from .physics import move


class Simulation:
    def __init__(self, world, algorithm: str, settings: Settings, seed: int):
        settings.validate()
        settings = deepcopy(settings)
        self.world, self.algorithm, self.settings, self.seed = world, algorithm, settings, seed
        self.robot = Robot(*world.start, radius=settings.robot_radius)
        self.robot.trajectory.append(world.start)
        self.controller = make_controller(algorithm, settings, seed)
        self.time = 0.0
        self.algorithm_runtime = 0.0
        self.reading = sense(self.robot, world, settings.sensor_range, settings.sensor_count)
        self.avoidance = Avoidance()
        self.latencies = []
        self.pending_danger = None
        self.was_dangerous = False
        self.danger_events = 0

    @property
    def done(self):
        return self.robot.reached_goal or self.time >= self.settings.max_time - 1e-8

    def apply(self, avoidance) -> None:
        danger = max(self.reading.proximity[:3]) >= self.settings.danger_threshold
        if danger and not self.was_dangerous:
            self.pending_danger = self.time
            self.danger_events += 1
        self.was_dangerous = danger
        self.avoidance = avoidance
        dt = self.settings.control_dt
        reaction = abs(avoidance.turn * self.settings.max_angular_speed) >= self.settings.response_turn_threshold or avoidance.brake >= 0.2
        if self.pending_danger is not None and reaction:
            self.latencies.append(self.time + dt - self.pending_danger)
            self.pending_danger = None
        speed, omega = motor_command(self.robot, self.world.goal, avoidance, self.settings)
        move(self.robot, self.world, speed, omega, dt, self.settings.goal_radius)
        self.time = round(self.time + dt, 9)
        self.reading = sense(self.robot, self.world, self.settings.sensor_range, self.settings.sensor_count)

    def step(self) -> None:
        if self.done:
            return
        start = time.perf_counter()
        avoidance = self.controller.step(self.reading)
        self.algorithm_runtime += time.perf_counter() - start
        self.apply(avoidance)

    def run(self, cancelled=lambda: False) -> None:
        if self.done:
            return
        network = self.controller.network
        if network is None:
            while not self.done and not cancelled():
                self.step()
            return
        # Single Brian2 run eliminates repeated before_run/code generation overhead.
        import brian2 as b2
        first = True
        non_neural_time = 0.0

        def control_tick():
            nonlocal first, non_neural_time
            if not first:
                avoidance = network.readout()
                start = time.perf_counter()
                self.apply(avoidance)
                non_neural_time += time.perf_counter() - start
            first = False
            if self.done or cancelled():
                network.net.stop()
            else:
                network.prepare(self.reading.proximity)

        operation = b2.NetworkOperation(control_tick, dt=self.settings.control_dt * b2.second, when="start", order=-10)
        network.net.add(operation)
        start = time.perf_counter()
        try:
            network.net.run((self.settings.max_time + self.settings.control_dt) * b2.second, namespace={})
        finally:
            self.algorithm_runtime += time.perf_counter() - start - non_neural_time
            network.net.remove(operation)
