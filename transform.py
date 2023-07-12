import jax.numpy as jnp

import jaxlie


def translate_wrench(wrench: jnp.ndarray, translation: jnp.ndarray) -> jnp.ndarray:
    """Compute a new wrench by translating the point of application."""
    return jnp.concatenate(
        [
            wrench[:3] + jnp.cross(translation, wrench[3:]),
            wrench[3:],
        ]
    )


def rotate_wrench(wrench: jnp.ndarray, rotation: jaxlie.SO3) -> jnp.ndarray:
    """Compute a new wrench by rotating the frame of reference."""
    return jnp.concatenate(
        [
            rotation * wrench[:3],
            rotation * wrench[3:],
        ]
    )


def transform_wrench(wrench: jnp.ndarray, transform: jaxlie.SE3) -> jnp.ndarray:
    """Compute a new wrench by applying a transform."""
    return translate_wrench(
        rotate_wrench(wrench, transform.rotation()),
        transform.translation,
    )
