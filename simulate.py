import jax
import jax_dataclasses as jdc
import jax.numpy as jnp
import jaxlie
import matplotlib.pyplot as plt

from descriptors import *
from dynamics import *
from visualizer import Visualizer

WIND_SPEED = 5.0
INITIAL_POSITION = jnp.array([0.0, 0.0, 3.0])
TARGET_TETHER_LENGTHS = 3.0


@jax.jit
def euler_step(x: State, x_dot: jnp.ndarray, dt: float) -> State:
    """Euler integration step."""
    return x + x_dot * dt


@jax.jit
def forward_dynamics(x: State, u: Control, params: Params) -> jnp.ndarray:
    # Compute the wrench on the kite in the kite frame
    wrench = compute_wrench(x, u, params)
    # Compute the acceleration of the kite in the kite frame
    kite_dot = apply_wrench(x.kite, wrench, params.mass, params.inertia.matrix())
    # There is no change in the wind state
    wind_dot = jnp.zeros(WindState.tangent_dim)

    return jnp.concatenate([kite_dot, wind_dot])


def compute_tether_jacobian(state, params):
    return jax.jacobian(compute_tether_lengths, 0)(state, params).kite.t @ state.kite.v


@jax.jit
def controller(state: State, params: Params) -> Control:
    kp = 1.0
    kd = 1.0
    l = compute_tether_lengths(state, params)
    l_dot = compute_tether_jacobian(state, params)
    return Control(kp * (l - TARGET_TETHER_LENGTHS) + kd * l_dot)


def simulate(initial_state, controller, params, dt, duration):
    """Simulate the kite dynamics forward in time."""
    # Set up initial state
    state = initial_state
    log = [state]
    # Iterate over the simulation duration
    for _ in range(int(duration / dt)):
        u = controller(state, params)
        # Compute the state derivative
        x_dot = forward_dynamics(state, u, params)
        if jnp.linalg.norm(x_dot) > 1e6:
            break
        # Integrate forward in time
        state = euler_step(state, x_dot, dt)
        # Log the state
        log.append(state)

    return log


if __name__ == "__main__":
    params = Params()
    # Set up initial state
    with jdc.copy_and_mutate(State.identity()) as state:
        state.kite.R = jaxlie.SO3.from_rpy_radians(0.1, 0.1, 0.1)
        state.wind.v = jnp.array([-WIND_SPEED, 0.0, 0.0])
        state.kite.t = INITIAL_POSITION

    log = simulate(state, controller, params, 0.001, 20.0)

    if len(log) < 100:
        data = jnp.stack([l.kite.local_velocity() for l in log], axis=0)
        plt.plot(data, label=["r", "p", "y", "x", "y", "z"])
        plt.legend()
        plt.show()

    else:
        vis = Visualizer().open()
        vis.add_kite(params)
        for state in log[::10]:
            vis.draw_state(state, rate=0.01)
