"""MultiOutput Kernels."""

from abc import abstractmethod

import jax.numpy as jnp
from jaxtyping import Num

from gpjax.kernels import AbstractKernel
from gpjax.typing import Array


# TODO describe various output shapes
class MultioutputKernel(AbstractKernel):
    """
    Multi Output Kernel class.

    This kernel can represent correlation between outputs of different datapoints.

    The `full_output_cov` argument holds whether the kernel should calculate
    the covariance between the outputs. In case there is no correlation but
    `full_output_cov` is set to True the covariance matrix will be filled with zeros
    until the appropriate size is reached.
    """

    @property
    @abstractmethod
    def num_latent_gps(self) -> int:
        """The number of latent GPs in the multioutput kernel"""
        raise NotImplementedError

    @property
    @abstractmethod
    def latent_kernels(self) -> tuple[AbstractKernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        raise NotImplementedError

    @abstractmethod
    def K(
        self,
        X: Num[Array, "N D"],
        X2: Num[Array, "M D"] | None = None,
        full_output_cov: bool = True,
    ) -> Array:
        """
        Returns the correlation of f(X) and f(X2), where f(.) can be multi-dimensional.

        :param X: data matrix
        :param X2: data matrix
        :param full_output_cov: calculate correlation between outputs.
        :return: cov[f(X), f(X2)]
        """
        raise NotImplementedError

    @abstractmethod
    def K_diag(self, X: Num[Array, "N D"], full_output_cov: bool = True) -> Array:
        """
        Returns the correlation of f(X) and f(X), where f(.) can be multi-dimensional.

        :param X: data matrix
        :param full_output_cov: calculate correlation between outputs.
        :return: var[f(X)]
        """
        raise NotImplementedError

    def __call__(
        self,
        X: Num[Array, "N D"],
        X2: Num[Array, "M D"] | None = None,
        *,
        full_cov: bool = False,
        full_output_cov: bool = True,
        presliced: bool = False,
    ) -> Array:
        if not presliced:
            X, X2 = self.slice(X, X2)
        if not full_cov and X2 is not None:
            raise ValueError(
                "Ambiguous inputs: passing in `X2` is not compatible with `full_cov=False`."
            )
        if not full_cov:
            return self.K_diag(X, full_output_cov=full_output_cov)
        return self.K(X, X2, full_output_cov=full_output_cov)


class SharedIndependent(MultioutputKernel):
    """
    - Shared: we use the same kernel for each latent GP
    - Independent: Latents are uncorrelated a priori.

    .. warning::
       This class is created only for testing and comparison purposes.
       Use `gpflow.kernels` instead for more efficient code.
    """

    def __init__(self, kernel: AbstractKernel, output_dim: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.output_dim = output_dim

    @property
    def num_latent_gps(self) -> int:
        # In this case number of latent GPs (L) == output_dim (P)
        return self.output_dim

    @property
    def latent_kernels(self) -> tuple[AbstractKernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return (self.kernel,)

    def K(
        self,
        X: Num[Array, "N D"],
        X2: Num[Array, "M D"] | None = None,
        full_output_cov: bool = True,
    ) -> Array:
        if X2 is None:
            K = self.kernel.gram(X)
        else:
            K = self.kernel.cross_covariance(X, X2)

        if full_output_cov:
            return jnp.tile(
                jnp.tile(K, (self.num_latent_gps, 1)), (1, self.num_latent_gps)
            )
        else:
            return jnp.kron(jnp.eye(self.num_latent_gps, dtype=int), K)

    def K_diag(self, X: Num[Array, "N D"], full_output_cov: bool = True) -> Array:
        K = self.kernel.diagonal(X)
        Ks = jnp.tile(K, (self.num_latent_gps, 1))
        return jnp.diag(Ks) if full_output_cov else Ks
