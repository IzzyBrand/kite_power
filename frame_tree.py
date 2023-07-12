from __future__ import annotations

from dataclasses import dataclass

from transform import rotate_spatial_velocity

import jax.numpy as jnp
import jaxlie


@dataclass
class Frame:
    # Position of frame origin in parent frame
    pose: jaxlie.SE3 = None
    # Velocity of frame origin in parent frame
    velocity: jnp.ndarray = None
    # Acceleration of frame origin in parent frame
    acceleration: jnp.ndarray = None
    # Parent frame
    relative_to: Frame = None


def compute_global_pose(frame: Frame) -> jaxlie.SE3:
    if frame.relative_to is None:
        return frame.pose
    else:
        return compute_global_pose(frame.relative_to) @ frame.pose


def compute_global_velocity(frame: Frame) -> jnp.ndarray:
    if frame.relative_to is None:
        return frame.velocity
    else:
        return compute_global_velocity(frame.relative_to) + rotate_spatial_velocity(
            compute_global_pose(frame.relative_to).rotation(), frame.velocity
        )


def compute_global_acceleration(frame: Frame) -> jnp.ndarray:
    if frame.relative_to is None:
        return frame.acceleration
    else:
        return compute_global_acceleration(frame.relative_to) + rotate_spatial_velocity(
            compute_global_pose(frame.relative_to).rotation(), frame.acceleration
        )

def compute_global_frame(frame: Frame) -> Frame:
    return Frame(
        pose=compute_global_pose(frame),
        velocity=compute_global_velocity(frame),
        acceleration=compute_global_acceleration(frame),
    )