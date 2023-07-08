import jax.numpy as jnp

from descriptors import *

def test_state_jit():
    """Define a simple function on BaseState and make sure we can jit it."""
    @jax.jit
    def f(state: BaseState):
        new_model = BaseModel(state.model.state_dimension * 2)
        new_x = jnp.concatenate([state.x, state.x])
        new_state = BaseState(new_model, new_x)
        return new_state

    state = BaseState(BaseModel(10), jnp.arange(10))
    print(state)
    print(f(state))


def test_derived_model_jit():
    @dataclass
    class CustomModel(BaseModel):
        state_dimension: int = 10
        custom_param: float


    @jax.jit
    def f(state: BaseState):
        new_model = CustomModel(3, 3.14)
        new_x = state.x[:CustomModel.state_dimension]
        new_state = BaseState(new_model, new_x)
        return new_state

    state = BaseState(BaseModel(10), jnp.arange(10))
    print(state)
    print(f(state))