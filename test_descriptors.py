import jax_dataclasses as jdc
import jax.numpy as jnp

from descriptors import *

def test_identity_constructors():
    State.identity()

def test_vectorizeable():
    v = State.identity().vector()
    State.from_vector(v)

def test_state_dim():
    state = State.identity()
    assert State.state_dim == state.vector().size
    assert State.tangent_dim == state.tangent().vector().size