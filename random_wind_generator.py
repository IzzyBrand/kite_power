import jax.numpy as jnp
import jax


class RandomWindGenerator:
    """Generates smoothly varying random wind vectors.

    Smoothness is achieved by integrating a series of vectors, where the 0th vector is updated with
    random noise, and each subsequent vector is updated based on the previous vector.
    """

    def __init__(self, dimension=3, update_rate=1.0, contraction_rate=1.0, levels=3):
        """Initialize the wind generator.

        Args:
            dimension: The dimension of the wind vector.
            update_rate: The rate at which random noise is added to the wind vector.
            contraction_rate: The rate at which the wind vector contracts.
            levels: The number of vectors to integrate.
        """
        self.dimension = dimension
        self.update_rate = update_rate
        self.contraction_rate = contraction_rate
        self.levels = levels

        self.vs = [jnp.zeros(self.dimension) for _ in range(self.levels)]

    def step(self, dt, prng_key):
        """Step the wind generator forward in time.

        Args:
            dt: The timestep.
            prng_key: The random number generator key.

        Returns:
            The wind vector.
        """
        # Update the the 0th vector with random noise
        self.vs[0] += self.update_rate * jax.random.normal(prng_key, (self.dimension,))

        for i in range(1, self.levels):
            # Update each vector based on it's rate of change
            self.vs[i] += self.vs[i - 1] * dt
            # Contact each vector by how aligned it is with the next vector
            self.vs[i - 1] *= 1.0 - dt * self.vs[i - 1].dot(self.vs[i])

        for i in range(self.levels):
            # Contact each vector by how long it is
            self.vs[i] *= 1.0 - dt * self.contraction_rate * jnp.linalg.norm(self.vs[i])

        return self.vs[-1]


if __name__ == "__main__":
    import plotly.express as px

    prng_key = jax.random.PRNGKey(0)

    wind_generator = RandomWindGenerator(dimension=3)
    wind = [wind_generator.step(0.01, k) for k in jax.random.split(prng_key, 1000)]
    wind = jnp.stack(wind)

    px.line(wind).show()
