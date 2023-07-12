import jax.numpy as jnp

import jaxlie


def translate_wrench(translation: jnp.ndarray, wrench: jnp.ndarray) -> jnp.ndarray:
    """Compute a new wrench by translating the point of application."""
    assert translation.shape == (3,)
    assert wrench.shape == (6,)

    return jnp.concatenate(
        [
            wrench[:3] + jnp.cross(translation, wrench[3:]),
            wrench[3:],
        ]
    )


def rotate_wrench(rotation: jaxlie.SO3, wrench: jnp.ndarray) -> jnp.ndarray:
    """Compute a new wrench by rotating the frame of reference."""
    assert wrench.shape == (6,)

    return jnp.concatenate(
        [
            rotation * wrench[:3],
            rotation * wrench[3:],
        ]
    )


def transform_wrench(transform: jaxlie.SE3, wrench: jnp.ndarray) -> jnp.ndarray:
    """Compute a new wrench by applying a transform."""
    assert wrench.shape == (6,)

    return translate_wrench(
        rotate_wrench(wrench, transform.rotation()),
        transform.translation,
    )


rotate_spatial_velocity = rotate_wrench
