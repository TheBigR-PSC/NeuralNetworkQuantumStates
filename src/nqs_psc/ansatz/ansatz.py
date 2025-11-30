# Import necessary libraries
import platform
import netket as nk
import numpy as np
from functools import partial

# jax and jax.numpy
import jax
import jax.numpy as jnp

# Flax for neural network models
import flax.linen as nn


# Mean Field Ansatz
class MF(nn.Module):
    @nn.compact
    def __call__(self, x):
        lam = self.param("lambda", nn.initializers.normal(), (1,), float)
        p = nn.log_sigmoid(lam * x)
        return 0.5 * jnp.sum(p, axis=-1)


# Jastrow Ansatz
class Jastrow(nn.Module):
    @nn.compact
    def __call__(self, x):
        n_sites = x.shape[-1]
        J = self.param("J", nn.initializers.normal(), (n_sites, n_sites), float)
        dtype = jax.numpy.promote_types(J.dtype, x.dtype)
        J = J.astype(dtype)
        x = x.astype(dtype)
        J_symm = J.T + J
        return jnp.einsum("...i,ij,...j", x, J_symm, x)


class BM(nn.Module):
    alpha: float

    @nn.compact
    def __call__(self, x):
        n_sites = x.shape[-1]
        n_hidden = int(self.alpha * n_sites)

        W = self.param(
            "W", nn.initializers.normal(), (n_sites, n_hidden), jnp.complex128
        )
        b = self.param("b", nn.initializers.normal(), (n_hidden,), jnp.complex128)

        return jnp.sum(jnp.log(jnp.cosh(jnp.matmul(x, W) + b)), axis=-1)
