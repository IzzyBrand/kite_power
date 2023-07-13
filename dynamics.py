"""Dynamics model for a two-string kite."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax.lax import atan2

from descriptors import *
from transform import translate_wrench


def get_airspeed(state: State) -> float:
    """Compute the airspeed of the kite."""
    return jnp.linalg.norm(state.wind.v - state.kite.v)


def get_body_frame_wind_velocity(state: State) -> jnp.ndarray:
    """Compute the wind velocity in the body frame."""
    return state.kite.R.inverse() @ (state.wind.v - state.kite.v)


def get_dynamic_pressure(state: State) -> float:
    """Compute the dynamic pressure."""
    return 0.5 * state.wind.rho * get_airspeed(state) ** 2


def get_angle_of_attack(state: State) -> float:
    """Compute the angle of attack."""
    v = get_body_frame_wind_velocity(state)
    return atan2(v[2], v[0])


def get_side_slip_angle(state: State) -> float:
    """Compute the side slip angle."""
    v = get_body_frame_wind_velocity(state)
    return atan2(v[1], v[0])


def compute_aerodynamic_coefficients(state: State, params: Params) -> jnp.ndarray:
    # I'm not entirely clear on what this term should be called, but it appears in the computation
    # of the aerodynamic coefficients in http://avionics.nau.edu.ua/files/doc/VisSim.doc/6dof.pdf
    # on page 9. I believe it has something to do with the fact that the aerodynamic coefficients
    # are meant to be non-dimensional. It also appears in equation (22) of
    # https://www.sciencedirect.com/science/article/pii/S0307904X17301798
    velocity_scale = jnp.array(
        [
            params.wingspan,
            params.chord,
            params.wingspan,
            0,
            0,
            0,
        ]
    )
    # Compose each of the coefficient matrix with the appropriate state vector
    return (
        # The drag coefficient doesn't depend on the state
        jnp.array([0, 0, 0, -params.drag_coefficient, 0, 0])
        # Coefficients that depend on the angle of attack
        + params.angle_of_attack_coefficients * get_angle_of_attack(state)
        # Coefficients that depend on the side slip angle
        + params.side_slip_angle_coefficients * get_side_slip_angle(state)
        # Coefficients that depend on the angular velocity (scaled by the velocity scale)
        + params.velocity_coefficients * state.kite.local_velocity() * velocity_scale
    )


def compute_aerodynamic_wrench(state: State, params: Params) -> jnp.ndarray:
    """Compute the aerodynamic wrench on the kite.

    Args:
        state: kite state

    Returns:
        aerodynamic wrench vector in the kite frame
    """

    # Compute the aerodynamic coefficients
    aerodynamic_coefficients = compute_aerodynamic_coefficients(state, params)

    # Compute the dynamic pressure
    dynamic_pressure = get_dynamic_pressure(state)

    # I'm not entirely clear on what this term should be called, but it appears in the computation
    # of the aerodynamic coefficients in http://avionics.nau.edu.ua/files/doc/VisSim.doc/6dof.pdf
    # on page 9. I believe it has something to do with the fact that the aerodynamic coefficients
    # are meant to be non-dimensional. It also appears in equation (9) of
    # https://onlinelibrary.wiley.com/doi/10.1002/we.2591
    angular_scale = jnp.array(
        [
            params.chord,
            params.chord,
            params.chord,
            1,
            1,
            1,
        ]
    )

    # Compute the aerodynamic wrench
    return (
        dynamic_pressure
        * params.surface_area
        * aerodynamic_coefficients
        * angular_scale
    )


def compute_tether_wrench(
    state: State, control: Control, params: Params
) -> jnp.ndarray:
    """Compute the wrench resulting from the tethers on the kite"""
    total_wrench = jnp.zeros(6)
    for i in range(2):
        # Comopute a vector from the kite attachment point to the anchor point
        # in the world frame
        attachment_point = state.kite.get_pose() @ params.tether_attachments[i]
        anchor_point = params.anchor_positions[i]
        tether_vector = anchor_point - attachment_point
        # The tether force in the world frame applied at the kite attachment point
        tether_force = tether_vector / jnp.linalg.norm(tether_vector) * control.tau[i]
        # The tether force in the kite frame applied at the kite attachment point
        tether_force = state.kite.R.inverse() @ tether_force
        # The wrench in the kite frame applied at the kite attachment point
        tether_wrench = jnp.concatenate([jnp.zeros(3), tether_force])
        # The wrench in the kite frame applied at the kite center of mass
        total_wrench += translate_wrench(attachment_point, tether_wrench)

    return total_wrench


def compute_coriolis_wrench(state: State, params: Params) -> jnp.ndarray:
    # Get the angular velocity of the kite in the local frame
    omega = state.kite.local_velocity()[:3]
    # Compute the coriolis torque
    tau = jnp.cross(omega, params.inertia.matrix() @ omega)
    # Return the wrench
    return jnp.concatenate([tau, jnp.zeros(3)])


def compute_gravity_wrench(state: State, params: Params) -> jnp.ndarray:
    F_g = jnp.array([0, 0, -params.gravity])
    # Transform the force into the kite frame and return the wrench
    return jnp.concatenate([jnp.zeros(3), state.kite.R.inverse() @ F_g])


def compute_wrench(state: State, control: Control, params: Params) -> jnp.ndarray:
    """Compute the wrench on the kite in the kite frame."""
    return (
        compute_aerodynamic_wrench(state, params)
        + compute_tether_wrench(state, control, params)
        + compute_coriolis_wrench(state, params)
        + compute_gravity_wrench(state, params)
    )


def rigid_body_acceleration(wrench: jnp.ndarray, m: float, I: jnp.ndarray):
    """Returns the 6-dimensional acceleration vector of a rigid body."""
    return jnp.concatenate([jnp.linalg.solve(I, wrench[:3]), wrench[3:] / m])


if __name__ == "__main__":
    state = State.identity()
    control = Control.identity()
    params = Params()

    print(compute_aerodynamic_wrench(state, params))
    print(compute_aerodynamic_coefficients(state, params))
    print(compute_tether_wrench(state, control, params))
