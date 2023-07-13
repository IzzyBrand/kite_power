from functools import cache

import jax_dataclasses as jdc
import jax.numpy as jnp


@jdc.pytree_dataclass
class Inertia:
    x: jdc.Static[float]
    y: jdc.Static[float]
    z: jdc.Static[float]
    xy: jdc.Static[float] = 0.0
    xz: jdc.Static[float] = 0.0
    yz: jdc.Static[float] = 0.0

    @cache
    def matrix(self) -> jnp.ndarray:
        return jnp.array(
            [
                [self.x, -self.xy, -self.xz],
                [-self.xy, self.y, -self.yz],
                [-self.xz, -self.yz, self.z],
            ]
        )
