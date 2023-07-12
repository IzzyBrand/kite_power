from __future__ import annotations


import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

###############################################################################
# SO3 lie group (rotation matrices)
###############################################################################


def x_rotation(theta: float) -> jnp.ndarray:
    return jnp.array(
        [
            [1, 0, 0],
            [0, jnp.cos(theta), -jnp.sin(theta)],
            [0, jnp.sin(theta), jnp.cos(theta)],
        ]
    )


def y_rotation(theta: float) -> jnp.ndarray:
    return jnp.array(
        [
            [jnp.cos(theta), 0, jnp.sin(theta)],
            [0, 1, 0],
            [-jnp.sin(theta), 0, jnp.cos(theta)],
        ]
    )


def z_rotation(theta: float) -> jnp.ndarray:
    return jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta), 0],
            [jnp.sin(theta), jnp.cos(theta), 0],
            [0, 0, 1],
        ]
    )


@jax.jit
def SO3_from_euler(euler: jnp.ndarray) -> jnp.ndarray:
    return z_rotation(euler[2]) @ y_rotation(euler[1]) @ x_rotation(euler[0])


@jax.jit
def SO3_hat(w: jnp.ndarray) -> jnp.ndarray:
    # A Micro Lie Theory (Example 3)
    wx, wy, wz = w
    S = jnp.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
    return S


@jax.jit
def SO3_vee(S: jnp.ndarray) -> jnp.ndarray:
    # A Micro Lie Theory (Example 3)
    w = jnp.array([S[2, 1], S[0, 2], S[1, 0]])
    return w


@jax.jit
def SO3_exp(w: jnp.ndarray) -> jnp.ndarray:
    # A Micro Lie Theory (Example 4)
    # Compute the magnitude of the rotation
    θ = jnp.linalg.norm(w)
    # Avoid division by zero
    θ = jnp.where(θ, θ, 1e-8)
    S = SO3_hat(w)
    R = jnp.eye(3) + S * jnp.sin(θ) / θ + S @ S * (1.0 - jnp.cos(θ)) / θ ** 2
    return R


@jax.jit
def SO3_Exp(S):
    # [1] Example 4
    w = SO3_vee(S)
    R = SO3_exp(w)
    return R


@jax.jit
def SO3_Log(R):
    # [1] Example 4
    θ = jnp.arccos(0.5 * (jnp.trace(R) - 1.0))
    S = θ * (R - R.T) / (2.0 * jnp.sin(θ))
    return S


@jax.jit
def SO3_log(R):
    # [1] Example 4
    S = SO3_Log(R)
    w = SO3_vee(S)
    return w


@jax.jit
def SO3_identity():
    return jnp.eye(3)


###############################################################################
# SE3 lie group (homogenous transforms)
###############################################################################


@jax.jit
def make_homogenous_transform(R: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Take a rotation matrix and a translation vector and return a homogeneous
    transformation matrix
        ┌      ┐
        │ R  t │
        │ 0  1 │
        └      ┘
    """
    return jnp.block([[R, t.squeeze()[:, None]], [jnp.zeros((1, 3)), 1]])


###############################################################################
# S3 lie group (quaternions)
###############################################################################


@register_pytree_node_class
class S3:
    def __init__(self, q):
        self.q = q

    @classmethod
    def identity(cls) -> S3:
        return cls(jnp.array([1.0, 0.0, 0.0, 0.0]))

    @classmethod
    def random(cls, prng_key) -> S3:
        q = jax.random.normal(prng_key, (4,))
        return cls(q / jnp.linalg.norm(q))

    @classmethod
    def from_SO3(cls, R) -> S3:
        assert R.shape == (3, 3)

        # Modified from:
        # > "Converting a Rotation Matrix to a Quaternion" from Mike Day
        # > https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/R-to-quat.pdf

        def case0(m):
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = jnp.array(
                [
                    m[2, 1] - m[1, 2],
                    t,
                    m[1, 0] + m[0, 1],
                    m[0, 2] + m[2, 0],
                ]
            )
            return t, q

        def case1(m):
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = jnp.array(
                [
                    m[0, 2] - m[2, 0],
                    m[1, 0] + m[0, 1],
                    t,
                    m[2, 1] + m[1, 2],
                ]
            )
            return t, q

        def case2(m):
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = jnp.array(
                [
                    m[1, 0] - m[0, 1],
                    m[0, 2] + m[2, 0],
                    m[2, 1] + m[1, 2],
                    t,
                ]
            )
            return t, q

        def case3(m):
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = jnp.array(
                [
                    t,
                    m[2, 1] - m[1, 2],
                    m[0, 2] - m[2, 0],
                    m[1, 0] - m[0, 1],
                ]
            )
            return t, q

        # Compute four cases, then pick the most precise one.
        # Probably worth revisiting this!
        case0_t, case0_q = case0(R)
        case1_t, case1_q = case1(R)
        case2_t, case2_q = case2(R)
        case3_t, case3_q = case3(R)

        cond0 = R[2, 2] < 0
        cond1 = R[0, 0] > R[1, 1]
        cond2 = R[0, 0] < -R[1, 1]

        t = jnp.where(
            cond0,
            jnp.where(cond1, case0_t, case1_t),
            jnp.where(cond2, case2_t, case3_t),
        )
        q = jnp.where(
            cond0,
            jnp.where(cond1, case0_q, case1_q),
            jnp.where(cond2, case2_q, case3_q),
        )

        # We can also choose to branch, but this is slower.
        # t, q = jax.lax.cond(
        #     R[2, 2] < 0,
        #     true_fun=lambda R: jax.lax.cond(
        #         R[0, 0] > R[1, 1],
        #         true_fun=case0,
        #         false_fun=case1,
        #         operand=R,
        #     ),
        #     false_fun=lambda R: jax.lax.cond(
        #         R[0, 0] < -R[1, 1],
        #         true_fun=case2,
        #         false_fun=case3,
        #         operand=R,
        #     ),
        #     operand=R,
        # )

        return cls(q * 0.5 / jnp.sqrt(t))

    def to_SO3(self) -> jnp.ndarray:
        norm = self.q @ self.q
        q = self.q * jnp.sqrt(2.0 / norm)
        q = jnp.outer(q, q)
        return jnp.array(
            [
                [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
                [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
                [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
            ]
        )

    @classmethod
    def exp(cls, w) -> S3:
        # A Micro Lie Theory (Example 5)
        # [1] Example 5
        θ = jnp.linalg.norm(w)
        # Handle the case where θ is zero
        s = jnp.where(θ, jnp.sin(0.5 * θ) / θ, 1)
        c = jnp.cos(0.5 * θ)
        return S3(jnp.concatenate([jnp.array([c]), s * w]))

    def log(self) -> jnp.ndarray:
        # Flip the quaternion as needed
        q = jnp.sign(self.q[0]) * self.q
        # And invert the exp operation
        θ = 2 * jnp.arccos(q[0])
        # Handle the case where θ is zero
        s = jnp.where(θ, jnp.sin(0.5 * θ) / θ, 1)
        return q[1:] / s

    def __mul__(self, other) -> S3:
        if isinstance(other, S3):
            w0, x0, y0, z0 = self.q
            w1, x1, y1, z1 = other.q
            return S3(
                jnp.array(
                    [
                        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                    ]
                )
            )

    def allclose(self, other: S3, **kwargs):
        """Checks equality between two elements of S3

        Note that two quaterions correspond to the same rotation when they
        are pointing in opposite directions."""
        return jnp.allclose(self.q, other.q, **kwargs) or jnp.allclose(
            self.q, -other.q, **kwargs
        )

    def tree_flatten(self):
        return (self.q,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])
