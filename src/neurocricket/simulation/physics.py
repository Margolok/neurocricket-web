"""Unicycle integration with substeps preventing high-speed tunnelling."""
import math
from .geometry import wrap_angle
from .robot import Robot
from .world import World


def move(robot: Robot, world: World, speed: float, omega: float, dt: float, goal_radius: float) -> None:
    if dt <= 0:
        raise ValueError("dt должен быть положительным")
    steps = max(1, math.ceil(abs(speed) * dt / (robot.radius / 3)), math.ceil(abs(omega) * dt / 0.05))
    hit = False
    travelled = 0.0
    for _ in range(steps):
        h = dt / steps
        heading = robot.theta + omega * h / 2
        x, y = robot.x + speed * math.cos(heading) * h, robot.y + speed * math.sin(heading) * h
        robot.theta = wrap_angle(robot.theta + omega * h)
        if world.collides(x, y, robot.radius):
            hit = True
            continue
        travelled += math.hypot(x - robot.x, y - robot.y)
        robot.x, robot.y = x, y
        if math.dist(robot.position, world.goal) <= goal_radius:
            robot.reached_goal = True
            break
    if hit and not robot.collided:
        robot.collisions += 1
    robot.collided = hit
    robot.speed, robot.angular_speed = travelled / dt, omega
    robot.path_length += travelled
    robot.trajectory.append(robot.position)
    robot.reached_goal |= math.dist(robot.position, world.goal) <= goal_radius
