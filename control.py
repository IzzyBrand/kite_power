import jax
import optax

from matplotlib import pyplot as plt

from descriptors import *
from dynamics import *


def pos_ddot(x: State, u: Control, params: Params):
    W = compute_wrench(x, u, params)
    x_dot = apply_wrench(x.kite, W, params.mass, params.inertia.matrix())
    return x_dot[-3:]


def dlen_dpos(x: KiteState, params: Params):
    return jax.jacobian(compute_tether_lengths, 0)(x, params).t


def len_ddot(x, u, params):
    return dlen_dpos(x.kite, params) @ pos_ddot(x, u, params)


@jax.jit
def cost(x, u, params):
    return jnp.linalg.norm(len_ddot(x, u, params))


@jax.jit
def dcost_du(x, u, params):
    return jax.value_and_grad(cost, 1)(x, u, params)


# @jax.jit
def solve_for_u(x, params, lr=0.1, max_iter=100):
    tau = Control.identity().tau
    for _ in range(max_iter):
        cost, grad = dcost_du(x, Control(tau), params)
        if cost < 1e-5:
            break
        else:
            tau = tau - lr * grad.tau

    print(cost)
    return Control(tau)


def solve_for_u_with_optax(x, u, p, lr=0.1, max_iter=10):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(u)

    for _ in range(max_iter):
        _, grad = dcost_du(x, u, p)
        updates, opt_state = optimizer.update(grad, opt_state, u)
        u = optax.apply_updates(u, updates)

    return u


def plot_cost(x, params):
    N = 100
    tau1 = jnp.linspace(8, 11, N)
    tau2 = jnp.linspace(8, 11, N)
    tau = jnp.stack(jnp.meshgrid(tau1, tau2), axis=-1).reshape(-1, 2)
    # print(tau.shape)

    def f(tau):
        return cost(x, Control(tau), params)

    costs = jax.vmap(f)(tau)
    min_cost = jnp.min(costs)
    min_tau = tau[jnp.argmin(costs)]
    print(min_cost, min_tau)

    plt.imshow(jax.vmap(f)(tau).reshape((N, N)))
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    params = Params()
    # Set up initial state
    with jdc.copy_and_mutate(State.identity()) as x:
        x.kite.R = jaxlie.SO3.from_rpy_radians(0.0, -0.3, 0.001)
        x.wind.v = jnp.array([-6.0, 0.0, 0.0])
        x.kite.t = jnp.array([-10.0, 0, 10.0])

    # cost(x, Control.identity(), params)
    # plot_cost(x, params)
    # u = Control.identity()
    u = solve_for_u_with_optax(x, params)
    print(u, cost(x, u, params))

    # print(dlen_du(state, Control(jnp.ones(2)), params))

    # print(pos_ddot(x, u, params))
    # print(len_ddot(x, u, params))
    # print(dcost_du(x, u, params))
