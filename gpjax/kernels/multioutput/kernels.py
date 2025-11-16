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


class SeparateIndependent(MultioutputKernel):
    """
    - Separate: we use different kernel for each output latent
    - Independent: Latents are uncorrelated a priori.
    """

    def __init__(self, kernels: Sequence[AbstractKernel]) -> None:
        super().__init__()
        self.kernels = kernels

    @property
    def num_latent_gps(self) -> int:
        # In this case number of latent GPs (L) == output_dim (P)
        return len(self.kernels)

    @property
    def latent_kernels(self) -> tuple[AbstractKernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    def K(
        self,
        X: Num[Array, "N D"],
        X2: Num[Array, "M D"] | None = None,
        full_output_cov: bool = True,
    ) -> Array:
        if X2 is None:
            if full_output_cov:
                Kxxs = jnp.stack([k.gram(X) for k in self.kernels], axis=-1)
            else:
                Kxxs = jnp.stack([k.gram(X) for k in self.kernels], axis=0)
        else:
            if full_output_cov:
                Kxxs = jnp.stack([k.cross_covariance(X, X2) for k in self.kernels], axis=-1)
            else:
                Kxxs = jnp.stack([k.cross_covariance(X, X2) for k in self.kernels], axis=0)
        return Kxxs

    def K_diag(self, X: Num[Array, "N D"], full_output_cov: bool = True) -> Array:
        Ks = jnp.stack([k.K_diag(X) for k in self.kernels], axis=-1)
        return jnp.diag(Ks) if full_output_cov else Ks


class LinearCoregionalization(MultioutputKernel):
    """
    Linear mixing of the latent GPs to form the output.
    """

    def __init__(self, kernels: Sequence[AbstractKernel], W: Num[Array, "Q L"]):
        super().__init__(self, kernels=kernels, name=name)
        self.W = W

    @property
    def num_latent_gps(self) -> int:
        return self.W.shape[-1]  # type: ignore[no-any-return]  # L

    @property
    def latent_kernels(self) -> Tuple[Kernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    @inherit_check_shapes
    def Kgg(self, X: Num[Array, "N D"], X2: Num[Array, "M D"] | None = None,) -> Array:
        if X2 is None:
            return jnp.stack([k.gram(X) for k in self.kernels], axis=0)
        else:
            return jnp.stack([k.cross_covariance(X, X2) for k in self.kernels], axis=0)

    def K(
        self,
        X: Num[Array, "N D"],
        X2: Num[Array, "M D"] | None = None,
        full_output_cov: bool = True,
    ) -> Array:
        Kxx = self.Kgg(X, X2)
    #     if X2 is None:
    #         cs(Kxx, "[L, batch..., N, N]")
    #         rank = tf.rank(X) - 1
    #         ones = tf.ones((rank + 1,), dtype=tf.int32)
    #         P = tf.shape(self.W)[0]
    #         L = tf.shape(self.W)[1]
    #         W_broadcast = cs(
    #             tf.reshape(self.W, tf.concat([[P, L], ones], 0)), "[P, L, broadcast batch..., 1, 1]"
    #         )
    #         KxxW = cs(Kxx[None, ...] * W_broadcast, "[P, L, batch..., N, N]")
    #         if full_output_cov:
    #             # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
    #             WKxxW = cs(tf.tensordot(self.W, KxxW, [[1], [1]]), "[P, P, batch..., N, N]")
    #             perm = tf.concat(
    #                 [
    #                     2 + tf.range(rank),
    #                     [0, 2 + rank, 1],
    #                 ],
    #                 0,
    #             )
    #             return cs(tf.transpose(WKxxW, perm), "[batch..., N, P, N, P]")
    #     else:
    #         cs(Kxx, "[L, batch..., N, batch2..., N2]")
    #         rank = tf.rank(X) - 1
    #         rank2 = tf.rank(X2) - 1
    #         ones12 = tf.ones((rank + rank2,), dtype=tf.int32)
    #         P = tf.shape(self.W)[0]
    #         L = tf.shape(self.W)[1]
    #         W_broadcast = cs(
    #             tf.reshape(self.W, tf.concat([[P, L], ones12], 0)),
    #             "[P, L, broadcast batch..., 1, broadcast batch2..., 1]",
    #         )
    #         KxxW = cs(Kxx[None, ...] * W_broadcast, "[P, L, batch..., N, batch2..., N2]")
    #         if full_output_cov:
    #             # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
    #             WKxxW = cs(
    #                 tf.tensordot(self.W, KxxW, [[1], [1]]), "[P, P, batch..., N, batch2..., N2]"
    #             )
    #             perm = tf.concat(
    #                 [
    #                     2 + tf.range(rank),
    #                     [0],
    #                     2 + rank + tf.range(rank2),
    #                     [1],
    #                 ],
    #                 0,
    #             )
    #             return cs(tf.transpose(WKxxW, perm), "[batch..., N, P, batch2..., N2, P]")
    #     # return tf.einsum('lnm,kl,kl->knm', Kxx, self.W, self.W)
    #     return tf.reduce_sum(W_broadcast * KxxW, axis=1)

    # @inherit_check_shapes
    # def K_diag(self, X: TensorType, full_output_cov: bool = True) -> tf.Tensor:
    #     K = cs(tf.stack([k.K_diag(X) for k in self.kernels], axis=-1), "[batch..., N, L]")
    #     rank = tf.rank(X) - 1
    #     ones = tf.ones((rank,), dtype=tf.int32)

    #     if full_output_cov:
    #         # Can currently not use einsum due to unknown shape from `tf.stack()`
    #         # return tf.einsum('nl,lk,lq->nkq', K, self.W, self.W)
    #         Wt = cs(tf.transpose(self.W), "[L, P]")
    #         L = tf.shape(Wt)[0]
    #         P = tf.shape(Wt)[1]
    #         return cs(
    #             tf.reduce_sum(
    #                 cs(K[..., None, None], "[batch..., N, L, 1, 1]")
    #                 * cs(tf.reshape(Wt, tf.concat([ones, [L, P, 1]], 0)), "[..., L, P, 1]")
    #                 * cs(tf.reshape(Wt, tf.concat([ones, [L, 1, P]], 0)), "[..., L, 1, P]"),
    #                 axis=-3,
    #             ),
    #             "[batch..., N, P, P]",
    #         )
    #     else:
    #         # return tf.einsum('nl,lk,lk->nkq', K, self.W, self.W)
    #         return cs(tf.linalg.matmul(K, self.W ** 2.0, transpose_b=True), "[batch..., N, P]")
