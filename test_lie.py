import jax
import jax.numpy as jnp

from lie import *

SEED = 0
NUM_TESTS = 5


def test_S3_SO3_conversions():
    prng_key = jax.random.PRNGKey(SEED)
    for subkey in jax.random.split(prng_key, NUM_TESTS):
        q = S3.random(subkey)

        R = q.to_SO3()
        # The transpose of a rotation matrix is it's inverse
        assert jnp.allclose(R.T, jnp.linalg.inv(R), atol=1e-5)

        q2 = S3.from_SO3(R)
        # The quaternion should have unit norm
        assert jnp.isclose(jnp.linalg.norm(q2.q), 1)

        R2 = q2.to_SO3()
        # The round-trip conversions should match
        assert jnp.allclose(R, R2, atol=1e-5)
        assert q.allclose(q2)

        R3 = S3(-q.q).to_SO3()
        # Negating the quaternion should result in the same rotation
        assert jnp.allclose(R, R3, atol=1e-5)


def test_SO3():
    prng_key = jax.random.PRNGKey(SEED)
    for subkey in jax.random.split(prng_key, NUM_TESTS):
        w = jax.random.uniform(subkey, (3,)) - 0.5
        S = SO3_hat(w)

        # Check that the round-trip conversions match
        assert jnp.allclose(w, SO3_vee(SO3_hat(w)))
        assert jnp.allclose(w, SO3_log(SO3_exp(w)))
        assert jnp.allclose(S, SO3_Log(SO3_Exp(S)))

        R = SO3_exp(w)
        # The transpose of a rotation matrix is it's inverse
        assert jnp.allclose(R.T, jnp.linalg.inv(R), atol=1e-5)


def test_S3_matches_SO3():
    prng_key = jax.random.PRNGKey(SEED)
    for subkey in jax.random.split(prng_key, NUM_TESTS):
        w = jax.random.uniform(subkey, (3,)) - 0.5
        assert jnp.allclose(S3.exp(w).to_SO3(), SO3_exp(w))

        q = S3.random(subkey)

        R = q.to_SO3()
        # TODO: The numerical stability is absolutely garbage here
        assert jnp.allclose(q.log(), SO3_log(R), atol=1e-3)


def test_S3():
    prng_key = jax.random.PRNGKey(SEED)
    for subkey in jax.random.split(prng_key, NUM_TESTS):
        q = S3.random(subkey)

        # Check that the round-trip conversions match
        q2 = S3.exp(q.log())
        assert q.allclose(q2, atol=1e-5)
