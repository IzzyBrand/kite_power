from dataclasses import dataclass, astuple

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import jax
from jax.interpreters.partial_eval import DynamicJaxprTracer

@dataclass
@register_pytree_node_class
class BaseModel:
    state_dimension: int

    def tree_flatten(self):
        return astuple(self), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@dataclass
@register_pytree_node_class
class BaseState:
    model: BaseModel
    x: jnp.ndarray

    def tree_flatten(self):
        return (self.model.tree_flatten()[0], self.x), self.model.__class__

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # print(cls, aux_data, children, children[0][0])
        # print(aux_data(*children[0]))
        return cls(aux_data.tree_unflatten(None, children[0]), children[1])


