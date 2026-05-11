import jax
import jax.numpy as jnp
import optax
import gpjax as gpx
from jax import random

# 1. SETUP SYNTHETIC DATA
P, Q, N = 6, 2, 40  # 6 outputs, rank 2 latent space, 40 points each
key = random.PRNGKey(123)
x_key, y_key, model_key = random.split(key, 3)

X = jnp.linspace(0, 5, N).reshape(-1, 1)

# Only Channel 0 and 1 have signal; others are pure noise
Y = jnp.zeros((N, P))
Y = Y.at[:, 0].set(jnp.sin(X).flatten())
Y = Y.at[:, 1].set(jnp.cos(X).flatten())
Y = Y + 0.05 * random.normal(y_key, (N, P))

# GPJax Multi-output Format: [X_values, channel_index]
X_tile = jnp.tile(X, (P, 1))
indices = jnp.repeat(jnp.arange(P), N).reshape(-1, 1)
X_multi = jnp.concatenate([X_tile, indices], axis=1)
Y_multi = Y.T.reshape(-1, 1)
dataset = gpx.Dataset(X=X_multi, y=Y_multi)

# 2. DEFINE MODEL
# active_dims=[0] ensures the RBF kernel ignores the index column
base_kernel = gpx.kernels.RBF(active_dims=[0])
coreg_matrix = gpx.parameters.CoregionalizationMatrix(num_outputs=P, rank=Q, key=jax.random.key(seed=42))
kernel = gpx.kernels.ICMKernel(base_kernel=base_kernel, coregionalization_matrix=coreg_matrix)

posterior = gpx.gps.Prior(
    mean_function=gpx.mean_functions.Zero(),
    kernel=kernel
) * gpx.likelihoods.MultiOutputGaussian(num_datapoints=N, num_outputs=P)

# 3. CUSTOM LOSS WITH GROUP LASSO
def loss_fn(params, data):
    # Standard Negative Log-Marginal Likelihood
    mll_func = gpx.objectives.conjugate_mll(posterior, data)
    neg_mll = -mll_func(params)
    
    # Access the weight matrix W (P x Q)
    W = params['kernel']['coregionalization_matrix']['W']
    
    # Group Lasso: L1 sum of row-wise L2 norms
    # This forces entire rows (channels) to zero
    lmbda = 15.0 
    row_norms = jnp.sqrt(jnp.sum(jnp.square(W), axis=1) + 1e-6)
    penalty = lmbda * jnp.sum(row_norms)
    
    return neg_mll + penalty

# 4. OPTIMIZATION
# params = gpx.parameters.initialise(posterior, model_key)
# optimizer = optax.adam(0.01)
# opt_state = optimizer.init(params)

# @jax.jit
# def step(params, opt_state, data):
#     loss, grads = jax.value_and_grad(loss_fn)(params, data)
#     updates, opt_state = optimizer.update(grads, opt_state, params)
#     return optax.apply_updates(params, updates), opt_state, loss

# Optimise GP's marginal log-likelihood using BFGS
opt_posterior, history = gpx.fit_scipy(
    model=posterior,
    objective=lambda p, d: -loss_fn(p, d),
    train_data=dataset,
)
# for i in range(1000):
#     params, opt_state, loss = step(params, opt_state, dataset)

# # 5. VERIFY SPARSITY
# W_final = params['kernel']['coregionalization_matrix']['W']
# norms = jnp.sqrt(jnp.sum(jnp.square(W_final), axis=1))

# print("Channel Importance (Row Norms of W):")
# for i, n in enumerate(norms):
#     print(f"Channel {i}: {n:.4f} {'[SIGNAL]' if n > 0.1 else '[REDUNDANT]'}")
