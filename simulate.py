import jax
import jax_dataclasses as jdc
import jax.numpy as jnp
import matplotlib.pyplot as plt

from descriptors import *
from dynamics import compute_wrench, rigid_body_acceleration

WIND_SPEED = 1.0
INITIAL_POSITION = jnp.array([0.0, 0.0, 2.0])


@jax.jit
def euler_step(x: State, x_dot: jnp.ndarray, dt: float) -> State:
    """Euler integration step."""
    return x + x_dot * dt


@jax.jit
def forward_dynamics(x: State, u: Control, params: Params) -> jnp.ndarray:
    # Compute the wrench on the kite in the kite frame
    wrench = compute_wrench(x, u, params)
    # Compute the acceleration of the kite in the kite frame
    kite_acceleration = rigid_body_acceleration(
        wrench, params.mass, params.inertia.matrix()
    )

    kite_velocity = x.kite.local_velocity()

    # There is no change in the wind state
    wind_dot = jnp.zeros(WindState.tangent_dim)

    return jnp.concatenate([kite_velocity, kite_acceleration, wind_dot])


def simulate(initial_state, control, params, dt, duration):
    """Simulate the kite dynamics forward in time."""
    # Set up initial state
    state = initial_state
    log = [state.tangent().vector()]
    # Iterate over the simulation duration
    for _ in range(int(duration / dt)):
        # Compute the state derivative
        x_dot = forward_dynamics(state, control, params)
        # Integrate forward in time
        state = euler_step(state, x_dot, dt)
        # Log the state
        log.append(state.tangent().vector())

    return jnp.stack(log, axis=0)


if __name__ == "__main__":
    # Set up initial state
    with jdc.copy_and_mutate(State.identity()) as state:
        state.wind.v = jnp.array([-WIND_SPEED, 0.0, 0.0])
        state.kite.t = INITIAL_POSITION

    control = Control(jnp.ones(2))

    data = simulate(state, control, Params(), 0.001, 0.25)
    plt.plot(data[:, :6], label=["r", "p", "y", "x", "y", "z"])
    plt.legend()
    plt.show()
