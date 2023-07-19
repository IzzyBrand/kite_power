from collections import namedtuple
import time

import numpy as onp
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import jaxlie
import jax_dataclasses as jdc
import matplotlib.pyplot as plt

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
        x.kite.v = jnp.array([-1.0, 0.0, 0.0])
        assert get_airspeed(x) == 2.0


def test_get_angle_of_attack():
    # Kite stationary -- direct headwind
    with jdc.copy_and_mutate(State.identity()) as x:
        x.wind.v = jnp.array([-1.0, 0.0, 0.0])
        print(get_angle_of_attack(x))
        assert jnp.isclose(get_angle_of_attack(x), 0.0)

    # Pitch the kite down
    theta = 0.1
    with jdc.copy_and_mutate(x) as x:
        x.kite.R = jaxlie.SO3.from_rpy_radians(0.0, theta, 0.0)
        assert jnp.isclose(get_angle_of_attack(x), -theta)

    # Reset the pitch, and move the kite up
    with jdc.copy_and_mutate(x) as x:
        x.kite.R = jaxlie.SO3.identity()
        x.kite.v = jnp.array([0.0, 0.0, 1.0])
        assert jnp.isclose(get_angle_of_attack(x), -0.25 * jnp.pi)


def test_get_side_slip_angle():
    # Kite stationary -- direct headwind
    with jdc.copy_and_mutate(State.identity()) as x:
        x.wind.v = jnp.array([-1.0, 0.0, 0.0])
        assert get_side_slip_angle(x) == 0.0

    # Yaw the kite to the left
    theta = 0.1
    with jdc.copy_and_mutate(x) as x:
        x.kite.R = jaxlie.SO3.from_rpy_radians(0.0, 0.0, theta)
        assert jnp.isclose(get_side_slip_angle(x), theta)

    # Reset the pitch and have it slide sideways
    with jdc.copy_and_mutate(x) as x:
        x.kite.R = jaxlie.SO3.identity()
        x.kite.v = jnp.array([0.0, 1.0, 0.0])
        assert jnp.isclose(get_side_slip_angle(x), -0.25 * jnp.pi)


def test_linear_and_angular_velocity_independent_under_integration():
    with jdc.copy_and_mutate(SingleRigidBodyState.identity()) as x1:
        x1.v = jnp.ones(3)
    with jdc.copy_and_mutate(x1) as x2:
        x2.omega = jnp.ones(3)

    for _ in range(10):
        x1 = x1 + apply_wrench(x1, jnp.zeros(6))
        x2 = x2 + apply_wrench(x2, jnp.zeros(6))

    # Make sure that both states translated
    assert not jnp.allclose(jnp.zeros(3), x1.t)
    assert not jnp.allclose(jnp.zeros(3), x2.t)
    # Only the second state rotated
    assert jnp.allclose(jnp.eye(3), x1.R.as_matrix())
    assert not jnp.allclose(jnp.eye(3), x2.R.as_matrix())
    # And their translations are the same
    assert jnp.allclose(x1.t, x2.t)


def test_pitch_only_aerodynamic_wrench():
    with jdc.copy_and_mutate(State.identity()) as x:
        x.kite.R = jaxlie.SO3.from_rpy_radians(0.0, jnp.radians(-30.0), 0.0)
        x.wind.v = jnp.array([-1.0, 0.0, 0.0])

    # Check that the kite's orientation results in the nose pointing up
    assert (x.kite.R @ jnp.array([1.0, 0.0, 0.0]))[2] > 0.0
    # And also verify that the angle of attack is positive, as excpected
    assert get_angle_of_attack(x) > 0.0

    # Compute the aerodynamic wrench and break out it's components. Note that the force component of
    # the wrench is expressed in the body frame.
    W = compute_aerodynamic_wrench(x, Params())
    roll, _, yaw = W[:3]
    drag, lateral, lift = x.kite.R @ W[3:]

    assert jnp.isclose(roll, 0.0)
    assert jnp.isclose(yaw, 0.0)
    assert jnp.isclose(lateral, 0.0)
    assert drag < 0.0
    assert lift > 0.0

    # If we fly backward, the draw and lift are appropriately flipped
    with jdc.copy_and_mutate(x) as x:
        x.wind.v = jnp.array([1.0, 0.0, 0.0])

    W = compute_aerodynamic_wrench(x, Params())
    drag, _, lift = x.kite.R @ W[3:]
    assert drag > 0.0
    assert lift < 0.0


def test_yaw_only_aerodynamic_wrench():
    with jdc.copy_and_mutate(State.identity()) as x:
        x.kite.R = jaxlie.SO3.from_rpy_radians(0.0, 0.0, jnp.radians(30.0))
        x.wind.v = jnp.array([-1.0, 0.0, 0.0])

    # Check that the kite's orientation results in the nose pointing to the left
    assert (x.kite.R @ jnp.array([1.0, 0.0, 0.0]))[1] > 0.0

    W = compute_aerodynamic_wrench(x, Params())
    drag, lateral, _ = x.kite.R @ W[3:]

    # Expect a lateral force in the direction opposite the nose
    assert lateral < 0.0
    # And there should be the usual drag
    assert drag < 0.0


def test_roll_and_pitch_aerodynamic_wrench():
    with jdc.copy_and_mutate(State.identity()) as x:
        x.kite.R = jaxlie.SO3.from_rpy_radians(
            jnp.radians(-30.0), jnp.radians(-30.0), 0.0
        )
        x.wind.v = jnp.array([-1.0, 0.0, 0.0])

    # Check that the kite's orientation results in the nose pointing up
    assert (x.kite.R @ jnp.array([1.0, 0.0, 0.0]))[2] > 0.0
    # And also that the left wingtip is pointing down
    assert (x.kite.R @ jnp.array([0.0, 1.0, 0.0]))[2] < 0.0

    W = compute_aerodynamic_wrench(x, Params())
    drag, lateral, lift = x.kite.R @ W[3:]

    # There should a lateral force to the left resulting from the lift component of the banked kite
    assert drag < 0.0
    assert lateral > 0.0
    assert lift > 0.0


def test_aerodynamic_rotational_damping():
    prng_key = jax.random.PRNGKey(0)
    for k in jax.random.split(prng_key, 10):
        with jdc.copy_and_mutate(State.identity()) as x:
            x.kite.omega = 10.0 * jnp.sign(jax.random.normal(k, (3,)))
            x.wind.v = jnp.array([-1e-6, 0, 0])

        W = compute_aerodynamic_wrench(x, Params())
        assert jnp.all(jnp.sign(W[:3]) == -jnp.sign(x.kite.omega))


def test_tethers_can_only_pull():
    u_pull = Control(jnp.ones(2))
    u_push = Control(-jnp.ones(2))

    with jdc.copy_and_mutate(State.identity()) as x:
        x.kite.t = jnp.array([10.0, 0.0, 0.0])
        assert compute_tether_wrench(x, u_pull, Params())[3] < 0.0
        assert compute_tether_wrench(x, u_push, Params())[3] == 0.0

    with jdc.copy_and_mutate(State.identity()) as x:
        x.kite.t = jnp.array([0.0, -10.0, 0.0])
        assert compute_tether_wrench(x, u_pull, Params())[4] > 0.0
        assert compute_tether_wrench(x, u_push, Params())[4] == 0.0

    with jdc.copy_and_mutate(State.identity()) as x:
        x.kite.t = jnp.array([0.0, 0.0, 10.0])
        assert compute_tether_wrench(x, u_pull, Params())[5] < 0.0
        assert compute_tether_wrench(x, u_push, Params())[5] == 0.0


def plot_angle_of_attack_and_side_slip_angle():
    def aoa(theta):
        with jdc.copy_and_mutate(State.identity()) as x:
            x.wind.v = jnp.array([-1.0, 0, 0])
            x.kite.R = jaxlie.SO3.from_rpy_radians(0, theta, 0)
            return get_angle_of_attack(x)

    def ssa(theta):
        with jdc.copy_and_mutate(State.identity()) as x:
            x.wind.v = jnp.array([-1.0, 0, 0])
            x.kite.R = jaxlie.SO3.from_rpy_radians(0, 0, theta)
            return get_side_slip_angle(x)

    theta = jnp.linspace(-4, 4, 100)
    plt.plot(theta, jax.vmap(aoa)(theta))
    plt.plot(theta, jax.vmap(ssa)(theta))
    plt.show()


# @jax.jit
# def forward_dynamics(x, params):
#     W_c = compute_coriolis_wrench(x.omega, params.I)
#     W_g = compute_gravity_wrench(x.R, params.m, params.g)
#     return apply_wrench(x, W_c + W_g, params.m, params.I)


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


if __name__ == "__main__":
    test_aerodynamic_rotational_damping()
#     # simulate_box_parabola()
#     simulate_tennis_racket_theorem()
