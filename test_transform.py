import jax.numpy as jnp
import jaxlie

from misc_math import skew
from transform import *


def test_translate_wrench_reversible():
    t = jnp.arange(3)
    W = jnp.arange(6)
    W2 = translate_wrench(t, W)
    W3 = translate_wrench(-t, W2)
    assert jnp.allclose(W, W3)


def test_rotate_wrench_reversible():
    R = jaxlie.SO3.from_rpy_radians(*jnp.arange(3))
    W = jnp.arange(6, dtype=jnp.float32)
    W2 = rotate_wrench(R, W)
    W3 = rotate_wrench(R.inverse(), W2)
    assert jnp.allclose(W, W3, atol=jaxlie.utils.get_epsilon(W.dtype))


def test_transform_wrench_reversible():
    T = jaxlie.SE3.exp(jnp.arange(6, dtype=jnp.float32))
    W = jnp.arange(6, dtype=jnp.float32)
    W2 = transform_wrench(T, W)
    W3 = transform_wrench(T.inverse(), W2)
    assert jnp.allclose(W, W3, atol=jaxlie.utils.get_epsilon(W.dtype))

]
def test_translate_wrench():
    point_of_application = jnp.array([1.0, 0.0, 0.0])
    # Pure force applied in the z direciton at the point of application
    W = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    # Compute the corresponding wrench at the origin
    wrench_at_origin = translate_wrench(point_of_application, W)

    assert jnp.allclose(wrench_at_origin, jnp.array([0.0, -1.0, 0.0, 0.0, 0.0, 1.0]))

    # Integrate the wrench's rotation component, and check that it rotates an
    # x-axis vector to a positive z-axis vector
    R = jaxlie.SO3.exp(wrench_at_origin[:3])
    v = R @ jnp.array([1.0, 0.0, 0.0])
    assert v[2] > 0.0


def test_screw_transform():
    R = jaxlie.SO3.from_rpy_radians(*jnp.arange(3))
    t = jnp.arange(3)
    X = jaxlie.SE3.from_rotation_and_translation(R, t)
    W = jnp.arange(6)

    A_R = jnp.block([[R.as_matrix(), jnp.zeros((3, 3))],
                     [jnp.zeros((3, 3)), R.as_matrix()]])
    A_t = jnp.block([[jnp.eye(3), skew(t)],
                     [jnp.zeros((3, 3)), jnp.eye(3)]])
    A_X = A_t @ A_R

    assert jnp.allclose(rotate_wrench(R, W), A_R @ W)
    assert jnp.allclose(translate_wrench(t, W), A_t @ W)
    assert jnp.allclose(transform_wrench(X, W), A_X @ W)
    assert jnp.allclose(screw_transform(R.as_matrix(), t), A_X)