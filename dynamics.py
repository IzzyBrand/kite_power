"""Dynamics model for a two-string kite."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.lax import atan2
from jax.tree_util import register_pytree_node_class
import jaxlie

from transform import translate_wrench


@dataclass
class KiteModel:

    state_dimension: int = 13

    mass: float = 4.0
    surface_area: float = 14.0
    chord: float = 1.5
    wingspan: float = 5.8

    drag_coefficient = 0.53

    # Aerodynamic coefficients as described in
    # http://avionics.nau.edu.ua/files/doc/VisSim.doc/6dof.pdf page 9

    angle_of_attack_coefficients = jnp.array([0, -0.7633, 0, 0.176, 0, -2.97])
    side_slip_angle_coefficients = jnp.array([-0.1, 0, -0.027, 0, -1.57, 0])

    # Note that the linear velocity coefficients are all zero because the majority of the
    # aerodynamic forces are captured by the angle of attack and side slip angle coefficients
    velocity_coefficients = jnp.array([-0.15, -0.165, -0.002, 0, 0, 0])

    # Positions of the tether attachments points in the kite frame
    tether_attachments = jnp.array(
        [
            [0, 2.9, -0.5],
            [0, -2.9, -0.5],
        ]
    )


# @register_pytree_node_class
class KiteState:
    def __init__(self, model, x):
        assert x.shape == (model.state_dimension,)
        self.model = model

        quat, self.position, self.angular_velocity, self.linear_velocity = jnp.split(
            x, [4, 7, 10]
        )
        self.orientation = jaxlie.SO3.from_quaternion_xyzw(quat)

    def get_orientation(self) -> jaxlie.SO3:
        return self.orientation

    def get_position(self) -> jnp.ndarray:
        return self.position

    def get_angular_velocity(self) -> jnp.ndarray:
        return self.angular_velocity

    def get_linear_velocity(self) -> jnp.ndarray:
        return self.linear_velocity

    def get_pose(self) -> jaxlie.SE3:
        return jaxlie.SE3.from_rotation_and_translation(
            self.get_orientation(), self.get_position()
        )

    def get_velocity(self) -> jnp.ndarray:
        return jnp.concatenate([self.angular_velocity, self.linear_velocity])

    def get_vector(self) -> jnp.ndarray:
        return jnp.concatenate(
            [
                self.orientation.q,
                self.position,
                self.angular_velocity,
                self.linear_velocity,
            ]
        )


# @register_pytree_node_class
class WindState:
    # Dimension of state vector
    size = 4

    def __init__(self, x):
        assert x.shape == (self.size,)

        self.velocity = x[:3]
        self.density = x[3]

    def get_velocity(self):
        return self.velocity

    def get_density(self):
        return self.density

    def get_vector(self) -> jnp.ndarray:
        return jnp.concatenate([self.velocity, jnp.ndarray([self.density])])


@dataclass
class TetherModel:
    anchor_position: jnp.ndarray


# @register_pytree_node_class
class TetherState:
    # Dimension of state vector
    size = 2

    def __init__(self, model, x):
        assert x.shape == (self.size,)
        self.model = model
        self.position, self.velocity = x

    def get_tension(self) -> float:
        # TODO: It seems that the control inputs -- motor torques -- will have
        # to be passed into the tether state
        return 1.0

    def get_vector(self) -> jnp.ndarray:
        jnp.ndarray([self.position, self.velocity])


class State:
    def __init__(
        self,
        kite: KiteState,
        wind: WindState,
        tethers: list[TetherState],
    ):
        self.kite = kite
        self.wind = wind
        self.tethers = tethers

        self.substates = [self.kite, self.wind] + self.tethers

    def get_airspeed(self) -> float:
        """Compute the airspeed of the kite."""
        return jnp.linalg.norm(
            self.wind.get_velocity() - self.kite.get_linear_velocity()
        )

    def get_body_frame_wind_velocity(self) -> jnp.ndarray:
        """Compute the wind velocity in the body frame."""
        return (
            self.kite.get_orientation()
            .inverse()
            .apply(self.wind.get_velocity() - self.kite.get_linear_velocity())
        )

    def get_dynamic_pressure(self) -> float:
        """Compute the dynamic pressure."""
        return 0.5 * self.wind.get_density() * self.get_airspeed() ** 2

    def get_angle_of_attack(self) -> float:
        """Compute the angle of attack."""
        v = self.get_body_frame_wind_velocity()
        return atan2(v[2], v[0])

    def get_side_slip_angle(self) -> float:
        """Compute the side slip angle."""
        v = self.get_body_frame_wind_velocity()
        return atan2(v[1], v[0])

    def get_vector(self) -> jnp.ndarray():
        return jnp.concatenate((s.get_vector() for s in self.substates))

    @classmethod
    def from_vector(cls, x):
        return cls(x)

    def tree_flatten(self):
        return (self.get_vector()), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls.from_vector(children[0])


def get_aerodynamic_coefficients(state: State) -> jnp.ndarray:
    # I'm not entirely clear on what this term should be called, but it appears in the computation
    # of the aerodynamic coefficients in http://avionics.nau.edu.ua/files/doc/VisSim.doc/6dof.pdf
    # on page 9. I believe it has something to do with the fact that the aerodynamic coefficients
    # are meant to be non-dimensional. It also appears in equation (22) of
    # https://www.sciencedirect.com/science/article/pii/S0307904X17301798
    velocity_scale = jnp.array(
        [
            state.kite.model.wingspan,
            state.kite.model.chord,
            state.kite.model.wingspan,
            0,
            0,
            0,
        ]
    )
    # Compose each of the coefficient matrix with the appropriate state vector
    return (
        # The drag coefficient doesn't depend on the state
        jnp.array([0, 0, 0, -state.kite.model.drag_coefficient, 0, 0])
        # Coefficients that depend on the angle of attack
        + state.kite.model.angle_of_attack_coefficients * state.get_angle_of_attack()
        # Coefficients that depend on the side slip angle
        + state.kite.model.side_slip_angle_coefficients * state.get_side_slip_angle()
        # Coefficients that depend on the angular velocity (scaled by the velocity scale)
        + state.kite.model.velocity_coefficients
        * state.kite.get_velocity()
        * velocity_scale
    )


# @jax.jit
def get_aerodynamic_wrench(state: State) -> jnp.ndarray:
    """Compute the aerodynamic wrench on the kite.

    Args:
        state: kite state

    Returns:
        aerodynamic wrench vector in the kite frame
    """

    # Compute the aerodynamic coefficients
    aerodynamic_coefficients = get_aerodynamic_coefficients(state)

    # Compute the dynamic pressure
    dynamic_pressure = state.get_dynamic_pressure()

    # I'm not entirely clear on what this term should be called, but it appears in the computation
    # of the aerodynamic coefficients in http://avionics.nau.edu.ua/files/doc/VisSim.doc/6dof.pdf
    # on page 9. I believe it has something to do with the fact that the aerodynamic coefficients
    # are meant to be non-dimensional. It also appears in equation (9) of
    # https://onlinelibrary.wiley.com/doi/10.1002/we.2591
    angular_scale = jnp.array(
        [
            state.kite.model.chord,
            state.kite.model.chord,
            state.kite.model.chord,
            1,
            1,
            1,
        ]
    )

    # Compute the aerodynamic wrench
    return (
        dynamic_pressure
        * state.kite.model.surface_area
        * aerodynamic_coefficients
        * angular_scale
    )


def get_tether_wrench(state: State) -> jnp.ndarray:
    """Compute the wrench resulting from the tethers on the kite"""
    total_wrench = jnp.zeros(6)
    for i, tether_state in enumerate(state.tethers):
        # Comopute a vector from the kite attachment point to the anchor point
        # in the world frame
        attachment_point = (
            state.kite.get_pose() @ state.kite.model.tether_attachments[i]
        )
        anchor_point = tether_state.model.anchor_position
        tether_vector = anchor_point - attachment_point
        # The tether force in the world frame applied at the kite attachment point
        tether_force = (
            tether_vector / jnp.linalg.norm(tether_vector) * tether_state.get_tension()
        )
        # The tether force in the kite frame applied at the kite attachment point
        tether_force = state.kite.get_orientation().inverse() @ tether_force
        # The wrench in the kite frame applied at the kite attachment point
        tether_wrench = jnp.concatenate([jnp.zeros(3), tether_force])
        # The wrench in the kite frame applied at the kite center of mass
        total_wrench += translate_wrench(attachment_point, tether_wrench)

    return total_wrench


if __name__ == "__main__":
    wind_x = jnp.array([1, 0.1, 0, 1.225])
    wind_state = WindState(wind_x)
    kite_x = jnp.concatenate([jaxlie.SO3.identity().as_quaternion_xyzw(), jnp.zeros(9)])
    kite_model = KiteModel()
    kite_state = KiteState(kite_model, kite_x)
    left_tether = TetherState(TetherModel(jnp.zeros(3)), jnp.zeros(2))
    right_tether = TetherState(TetherModel(jnp.zeros(3)), jnp.zeros(2))
    state = State(kite_state, wind_state, [left_tether, right_tether])

    print(get_aerodynamic_wrench(state))
    print(get_aerodynamic_coefficients(state))
    print(get_tether_wrench(state))
