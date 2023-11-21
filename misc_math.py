import jaxlie
import jax.numpy as jnp

skew = jaxlie._se3._skew

def block_diag(matrices):
    # Given a stack of k, nxm matrices, return a single (kn)x(kn) matrix
    k, n, m = matrices.shape
    return jnp.block([[matrices[i] if i == j else jnp.zeros((n, m)) for j in range(k)] for i in range(k)])
