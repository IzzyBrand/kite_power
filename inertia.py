from __future__ import annotations

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

    def matrix(self) -> jnp.ndarray:
        return jnp.array(
            [
                [self.x, self.xy, self.xz],
                [self.xy, self.y, self.yz],
                [self.xz, self.yz, self.z],
            ]
        )

    @classmethod
    def from_box_dimensions(cls, m, x, y, z):
        return cls(
            x=m * (y ** 2 + z ** 2) / 12,
            y=m * (x ** 2 + z ** 2) / 12,
            z=m * (x ** 2 + y ** 2) / 12,
        )
