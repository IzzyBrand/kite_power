import jax.numpy as jnp

import jaxlie
from misc_math import skew


def screw_transform(R, t):
    return jnp.block([
        [R, skew(t) @ R],
        [jnp.zeros((3, 3)), R]
    ])

# def spatial_motion_transform(R, t):
#     return jnp.block([
#         [R, jnp.zeros((3, 3))],
#         [skew(t) @ R, R]
#     ])


def translate_wrench(translation: jnp.ndarray, wrench: jnp.ndarray) -> jnp.ndarray:
    """Compute a new wrench by translating the point of application."""
    assert translation.shape == (3,)
    assert wrench.shape == (6,)

    return jnp.concatenate(
        [
            # TODO: figure out if this needs to be positive or negative
            wrench[:3] + jnp.cross(translation, wrench[3:]),
            wrench[3:],
        ]
    )


def rotate_wrench(rotation: jaxlie.SO3, wrench: jnp.ndarray) -> jnp.ndarray:
    """Compute a new wrench by rotating the frame of reference."""
    assert wrench.shape == (6,)

    return jnp.concatenate(
        [
            rotation @ wrench[:3],
            rotation @ wrench[3:],
        ]
    )


def transform_wrench(transform: jaxlie.SE3, wrench: jnp.ndarray) -> jnp.ndarray:
    """Compute a new wrench by applying a transform."""
    assert wrench.shape == (6,)

    return translate_wrench(
        transform.translation(),
        rotate_wrench(transform.rotation(), wrench),
    )


rotate_spatial_velocity = rotate_wrench
