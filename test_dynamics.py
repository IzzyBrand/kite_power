from collections import namedtuple
import time

import numpy as onp
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import jaxlie
import meshcat

from inertia import Inertia
from descriptors import SingleRigidBodyState
from dynamics import (
    compute_coriolis_wrench,
    compute_gravity_wrench,
    rigid_body_acceleration,
)


@jax.jit
def forward_dynamics(x, params):
    W_c = compute_coriolis_wrench(x.omega, params.I)
    W_g = compute_gravity_wrench(x.R, params.m, params.g)
    a = rigid_body_acceleration(W_c + W_g, params.m, params.I)
    x_dot = jnp.concatenate([x.velocity(), a])
    return x_dot


@jax.jit
def euler_step(x, x_dot, dt=0.003):
    return x + x_dot * dt


def run_simulation(
    x: SingleRigidBodyState, params, duration=1.0, dt=0.001, logging_rate=1.0 / 30.0
):
    log = [(0.0, x)]
    for t in onp.linspace(0, duration, int(duration / dt)):
        x_dot = forward_dynamics(x, params)
        x = euler_step(x, x_dot)
        if t % logging_rate < dt:
            log.append((t, x))

    return log


def visualize_log(log, playback_rate=1.0):
    vis = meshcat.Visualizer().open()
    vis["box"].set_object(meshcat.geometry.Box(onp.array([0.3, 0.7, 1.0])))
    t0 = time.time()
    for t, x in log:
        vis["box"].set_transform(onp.array(x.pose().as_matrix(), dtype=onp.float64))
        t_remaining = t0 + playback_rate * t - time.time()
        if t_remaining > 0:
            time.sleep(t_remaining)


def simulate_box_parabola():
    size = onp.array([0.3, 0.7, 1.0])
    m = 10.0
    I = Inertia.from_box_dimensions(m, *size).matrix()

    x = SingleRigidBodyState(
        R=jaxlie.SO3.identity(),
        t=jnp.zeros(3),
        omega=jnp.zeros(3),
        v=jnp.array([0, 0.5, 10.0]),
    )

    params = namedtuple("Params", ["m", "I", "g"])(m, I, 9.81)

    log = run_simulation(x, params, duration=3.0)
    visualize_log(log)


def simulate_tennis_racket_theorem():
    size = onp.array([0.3, 0.7, 1.0])
    m = 10.0
    I = Inertia.from_box_dimensions(m, *size).matrix()

    x = SingleRigidBodyState(
        R=jaxlie.SO3.identity(),
        t=jnp.zeros(3),
        omega=jnp.array([1e-4, 3.0, 0.0]),
        v=jnp.zeros(3),
    )

    params = namedtuple("Params", ["m", "I", "g"])(m, I, 0.0)

    log = run_simulation(x, params, duration=10.0)
    visualize_log(log)


if __name__ == "__main__":
    # simulate_box_parabola()
    simulate_tennis_racket_theorem()
