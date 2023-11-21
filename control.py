import jax
import optax

from matplotlib import pyplot as plt

from descriptors import *
from dynamics import *
from misc_math import *
from transform import *


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


def spatial_inertia_inverse(I, m):
    return jnp.block([
        [jnp.linalg.inv(I), jnp.zeros((3, 3))],
        [jnp.zeros((3, 3)), 1.0/m * jnp.eye(3)],
    ])




def compute_optimal_control(x:State, params: Params):
    # Compute the wrench on the kite that results with no control input
    # W = compute_wrench(x, Control.identity(), params)
    N = params.tether_attachments.shape[0]
    # Control vector (tether tensions)
    tau = jnp.arange(N) + 1
    # Tether vectors
    V_t = compute_tether_vectors(x.kite, params)
    V_t = -V_t / jnp.linalg.norm(V_t, axis=1, keepdims=True)
    V_t = jnp.concatenate([jnp.zeros([N,3]), V_t], axis=1)
    V_t = jnp.expand_dims(V_t, axis=-1)
    V_t = block_diag(V_t)
    # Transform tether wrenches to kite frame
    X_t = jax.vmap(screw_transform, in_axes=(None, 0))(x.kite.R.inverse().as_matrix(), -params.tether_attachments)
    X_t = block_diag(X_t)
    # Sum tether wrenches
    S = jnp.concatenate([jnp.eye(6)] * tau.size, axis=1)
    I_inv = spatial_inertia_inverse(params.inertia.matrix(), params.mass)
    # [6x6] x [6x6n] [6nx6n] x [6nxn] x [n]
    A = I_inv @ S @ X_t @ V_t

    A_state = I_inv @ compute_wrench(x, Control.identity(), params)

    J_l = jnp.concatenate([jnp.zeros([N, 3]), dlen_dpos(x.kite, params) @ x.kite.R.as_matrix()], axis=1)

    tau_opt = jnp.linalg.solve(J_l @ A, J_l @ A_state)
    print(tau_opt)


    # W1 = compute_tether_wrench(x, Control(tau), params)
    # W2 = A @ tau
    # print(W1)
    # print(W2)



if __name__ == "__main__":
    params = Params()
    # Set up initial state
    with jdc.copy_and_mutate(State.identity()) as x:
        x.kite.R = jaxlie.SO3.from_rpy_radians(0.2, -0.3, 0.1)
        x.wind.v = jnp.array([-6.0, 0.0, 0.0])
        x.kite.t = jnp.array([-10.0, 0, 10.0])


    compute_optimal_control(x, params)
    # W = compute_wrench(x, Control.identity(), params)
    # I_inv = spatial_inertia_inverse(params.inertia.matrix(), params.mass)

    # acc = I_inv @ W

    # X = screw_transform(jnp.eye(3), -params.tether_attachments[0])

    # print(acc)
    # print(X@acc)

    # cost(x, Control.identity(), params)
    plot_cost(x, params)
    # u = Control.identity()
    u = solve_for_u_with_optax(x, Control.identity(), params)
    print(u, cost(x, u, params))

    # print(dlen_du(state, Control(jnp.ones(2)), params))

    # print(pos_ddot(x, u, params))
    # print(len_ddot(x, u, params))
    # print(dcost_du(x, u, params))
