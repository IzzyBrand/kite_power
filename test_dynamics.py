from collections import namedtuple
import time

import numpy as onp
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import jaxlie
import jax_dataclasses as jdc
import meshcat

from inertia import Inertia
from descriptors import SingleRigidBodyState, State
from dynamics import *


def test_get_airspeed():
    # If only the kite is moving, the airspeed is as expected
    with jdc.copy_and_mutate(State.identity()) as x:
        x.kite.v = jnp.array([1.0, 0.0, 0.0])
        assert get_airspeed(x) == 1.0

    # If the kite and the wind are moving at the same velocity, the airspeed is zero
    with jdc.copy_and_mutate(x) as x:
        x.wind.v = jnp.array([1.0, 0.0, 0.0])
        assert get_airspeed(x) == 0.0

    # If the kite and the wind are moving in opposite directions, the airspeed is double
    with jdc.copy_and_mutate(x) as x:
        x.kite.R = jaxlie.SO3.from_rpy_radians(0.0, 0.0, jnp.pi)
        assert get_airspeed(x) == 2.0


def test_get_angle_of_attack():
    # Kite stationary -- direct headwind
    with jdc.copy_and_mutate(State.identity()) as x:
        x.wind.v = jnp.array([-1.0, 0.0, 0.0])
        assert get_angle_of_attack(x) == 0.0

    # Pitch the kite up
    theta = 0.1
    with jdc.copy_and_mutate(x) as x:
        x.kite.R = jaxlie.SO3.from_rpy_radians(0.0, theta, 0.0)
        assert jnp.isclose(get_angle_of_attack(x), theta)

    # Pitch the kite down, and now it's moving
    with jdc.copy_and_mutate(x) as x:
        x.kite.R = jaxlie.SO3.from_rpy_radians(0.0, -theta, 0.0)
        x.kite.v = jnp.array([1.0, 0.0, 0.0])
        assert jnp.isclose(get_angle_of_attack(x), -0.5 * theta)


def test_get_side_slip_angle():
    # Kite stationary -- direct headwind
    with jdc.copy_and_mutate(State.identity()) as x:
        x.wind.v = jnp.array([-1.0, 0.0, 0.0])
        assert get_side_slip_angle(x) == 0.0

    # Pitch the kite up
    theta = 0.1
    with jdc.copy_and_mutate(x) as x:
        x.kite.R = jaxlie.SO3.from_rpy_radians(0.0, 0.0, -theta)
        assert jnp.isclose(get_side_slip_angle(x), theta)

    # Pitch the kite down, and now it's moving
    with jdc.copy_and_mutate(x) as x:
        x.kite.R = jaxlie.SO3.from_rpy_radians(0.0, 0.0, theta)
        x.kite.v = jnp.array([1.0, 0.0, 0.0])
        assert jnp.isclose(get_side_slip_angle(x), -0.5 * theta)


def test_linear_and_angular_velocity_independent_under_integration():
    with jdc.copy_and_mutate(SingleRigidBodyState.identity()) as x1:
        x1.v = jnp.ones(3)
    with jdc.copy_and_mutate(x1) as x2:
        x2.omega = jnp.ones(3)

    for _ in range(10):
        x1 = x1 + compute_single_rigid_body_x_dot(x1.velocity())
        x2 = x2 + compute_single_rigid_body_x_dot(x2.velocity())

    # Make sure that both states translated
    assert not jnp.allclose(jnp.zeros(3), x1.t)
    assert not jnp.allclose(jnp.zeros(3), x2.t)
    # Only the second state rotated
    assert jnp.allclose(jnp.eye(3), x1.R.as_matrix())
    assert not jnp.allclose(jnp.eye(3), x2.R.as_matrix())
    # And their translations are the same
    assert jnp.allclose(x1.t, x2.t)


# @jax.jit
# def forward_dynamics(x, params):
#     W_c = compute_coriolis_wrench(x.omega, params.I)
#     W_g = compute_gravity_wrench(x.R, params.m, params.g)
#     return compute_single_rigid_body_x_dot(x.velocity(), W_c + W_g, params.m, params.I)


# @jax.jit
# def euler_step(x, x_dot, dt=0.003):
#     return x + x_dot * dt


# def run_simulation(
#     x: SingleRigidBodyState, params, duration=1.0, dt=0.001, logging_rate=1.0 / 30.0
# ):
#     log = [(0.0, x)]
#     for t in onp.linspace(0, duration, int(duration / dt)):
#         x_dot = forward_dynamics(x, params)
#         x = euler_step(x, x_dot)
#         if t % logging_rate < dt:
#             log.append((t, x))

#     return log


# def visualize_log(log, playback_rate=1.0):
#     vis = meshcat.Visualizer().open()
#     vis["box"].set_object(meshcat.geometry.Box(onp.array([0.3, 0.7, 1.0])))
#     t0 = time.time()
#     for t, x in log:
#         vis["box"].set_transform(onp.array(x.pose().as_matrix(), dtype=onp.float64))
#         t_remaining = t0 + playback_rate * t - time.time()
#         if t_remaining > 0:
#             time.sleep(t_remaining)


# def simulate_box_parabola():
#     size = onp.array([0.3, 0.7, 1.0])
#     m = 10.0
#     I = Inertia.from_box_dimensions(m, *size).matrix()

#     x = SingleRigidBodyState(
#         R=jaxlie.SO3.identity(),
#         t=jnp.zeros(3),
#         omega=jnp.zeros(3),
#         v=jnp.array([0, 0.5, 10.0]),
#     )

#     params = namedtuple("Params", ["m", "I", "g"])(m, I, 9.81)

#     log = run_simulation(x, params, duration=3.0)
#     visualize_log(log)


# def simulate_tennis_racket_theorem():
#     size = onp.array([0.3, 0.7, 1.0])
#     m = 10.0
#     I = Inertia.from_box_dimensions(m, *size).matrix()

#     x = SingleRigidBodyState(
#         R=jaxlie.SO3.identity(),
#         t=jnp.zeros(3),
#         omega=jnp.array([1e-4, 3.0, 0.0]),
#         v=jnp.zeros(3),
#     )

#     params = namedtuple("Params", ["m", "I", "g"])(m, I, 0.0)

#     log = run_simulation(x, params, duration=10.0)
#     visualize_log(log)


# if __name__ == "__main__":
#     simulate_box_parabola()
#     # simulate_tennis_racket_theorem()
