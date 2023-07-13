import jax.numpy as jnp
import jaxlie

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
