import abc
from typing import Annotated, TypeVar
import jax_dataclasses as jdc

import jaxlie
from jaxlie.manifold._tree_utils import _map_group_trees
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from inertia import Inertia


@jdc.pytree_dataclass
class Params:
    gravity: float = 9.81

    # Kite inertial properties
    mass: float = 4.0
    inertia = Inertia(x=8.68, y=2.43, z=8.4, xz=0.33)

    # Kite geometry
    surface_area: float = 12.0
    chord: float = 2
    wingspan: float = 6

    # Aerodynamic coefficients as described in
    # http://avionics.nau.edu.ua/files/doc/VisSim.doc/6dof.pdf page 9
    drag_coefficient = 0.53
    angle_of_attack_coefficients = jnp.array([0, -0.7633, 0, 0.176, 0, -2.97])
    side_slip_angle_coefficients = jnp.array([-0.1, 0, -0.027, 0, -1.57, 0])
    # Note that the linear velocity coefficients are all zero because the majority of the
    # aerodynamic forces are captured by the angle of attack and side slip angle coefficients
    velocity_coefficients = jnp.array([-0.15, -0.165, -0.002, 0, 0, 0])

    # Positions of the tether attachments points in the kite frame
    tether_attachments = jnp.array(
        [
            [0, 3.0, -0.5],
            [0, -3.0, -0.5],
        ]
    )
    # Positions of the tether attachments in the world frame
    anchor_positions = jnp.zeros((2, 3))


class ManifoldObjectBase(abc.ABC):
    """Base class for states and control inputs requires classes to implement an `identity` method
    that returns an instance of the class"""

    @classmethod
    @abc.abstractmethod
    def identity(cls):
        pass


# Define a type variable that is a subclass of ManifoldObjectBase
ManifoldObjectType = TypeVar("ManifoldObjectType", bound=ManifoldObjectBase)


def vectorizeable(cls: ManifoldObjectType) -> ManifoldObjectType:
    """Decorator that adds vectorization methods to a state class."""

    cls_instance = cls.identity()
    vec, cls.from_vector = ravel_pytree(cls_instance)
    cls.vector = lambda self: ravel_pytree(self)[0]
    cls.state_dim = vec.size

    tangent_instance = jaxlie.manifold.zero_tangents(cls_instance)
    cls.tangent = lambda self: _map_group_trees(lambda x: x.log(), lambda x: x, self)
    tangent_vec, from_tangent_vec = ravel_pytree(tangent_instance)
    cls.tangent_dim = tangent_vec.size

    def __add__(self, other):
        return jaxlie.manifold.rplus(self, from_tangent_vec(other))

    cls.__add__ = __add__

    return cls


@vectorizeable
@jdc.pytree_dataclass
class SingleRigidBodyState(ManifoldObjectBase):
    # Rotation
    R: jaxlie.SO3
    # Translation
    t: Annotated[jnp.ndarray, (3,)]
    # Angular velocity in the kite frame
    omega: Annotated[jnp.ndarray, (3,)]
    # Linear velocity in the kite frame
    v: Annotated[jnp.ndarray, (3,)]

    def pose(self) -> jaxlie.SE3:
        return jaxlie.SE3.from_rotation_and_translation(self.R, self.t)

    def velocity(self) -> jnp.ndarray:
        return jnp.concatenate([self.omega, self.v])

    @classmethod
    def identity(cls):
        return cls(
            R=jaxlie.SO3.identity(),
            t=jnp.zeros(3),
            omega=jnp.zeros(3),
            v=jnp.zeros(3),
        )


KiteState = SingleRigidBodyState


@vectorizeable
@jdc.pytree_dataclass
class WindState(ManifoldObjectBase):
    v: Annotated[jnp.ndarray, (3,)]
    rho: Annotated[jnp.ndarray, (1,)]

    @classmethod
    def identity(cls):
        return cls(
            v=jnp.zeros(3),
            rho=jnp.array([1.225]),
        )


@vectorizeable
@jdc.pytree_dataclass
class State(ManifoldObjectBase):
    kite: KiteState
    wind: WindState

    @classmethod
    def identity(cls):
        return cls(
            kite=KiteState.identity(),
            wind=WindState.identity(),
        )


@vectorizeable
@jdc.pytree_dataclass
class Control(ManifoldObjectBase):
    tau: Annotated[jnp.ndarray, (2,)]

    @classmethod
    def identity(cls):
        return cls(
            tau=jnp.zeros(2),
        )
