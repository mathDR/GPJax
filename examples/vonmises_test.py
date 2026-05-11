#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Enable Float64 for more stable matrix inversions.
import blackjax
import jax
import equinox as eqx
import numpyro.distributions as npd
from jax import config
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import paramax
import optax

from examples.utils import use_mpl_style, clean_legend

config.update("jax_enable_x64", True)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


key = jr.key(123)

cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]


# In[2]:


n = 50
t = jr.uniform(key, shape=(n, 1), minval=0, maxval=5.0)
xtest = jnp.linspace(0, 5.0, 500).reshape(-1, 1)


# In[3]:


loc = jnp.sin(2.*jnp.pi*t)
kappa = 10


# In[4]:


vm = npd.VonMises(loc=loc,concentration=kappa).sample(key=jr.key(seed=42))


# In[5]:


loc.shape, t.shape


# In[6]:


plt.scatter(t,vm)


# In[7]:


kernel = gpx.kernels.RBF()  # 1-dimensional input
meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)


# In[8]:


D = gpx.Dataset(X=t, y=vm)


# In[9]:


likelihood = gpx.likelihoods.VonMises(1.5,num_datapoints=D.n)


# In[10]:


posterior = prior * likelihood


# In[11]:


print(type(posterior))


# # Optimization

# In[16]:


opt_posterior, history = gpx.fit(
    model=opt_posterior,
    # we use the negative lpd as we are minimising
    objective=lambda p, d: -gpx.objectives.log_posterior_density(p, d),
    train_data=D,
    optim=optax.adamw(learning_rate=0.01),
    num_iters=50000,
    key=key,
)


# In[17]:


map_latent_dist = opt_posterior.predict(xtest, train_data=D)
predictive_dist = opt_posterior.likelihood(map_latent_dist)


# In[18]:


plt.plot(xtest,predictive_dist.loc)


# In[ ]:





# # MCMC

# In[12]:


# Adapted from BlackJax's introduction notebook.
num_adapt = 1000
num_samples = 500

params, static = eqx.partition(posterior, eqx.is_array)


def logprob_fn(params):
    model = eqx.combine(params, static)
    model = paramax.unwrap(model)
    return gpx.objectives.log_posterior_density(model, D)

step_size = 1e-3
n_params = sum(jnp.size(leaf) for leaf in jtu.tree_leaves(params))
inverse_mass_matrix = jnp.ones(n_params)
nuts = blackjax.nuts(logprob_fn, step_size, inverse_mass_matrix)

state = nuts.init(params)

step = jax.jit(nuts.step)


def one_step(state, rng_key):
    state, info = step(rng_key, state)
    return state, (state, info)

keys = jax.random.split(key, num_samples)
_, (states, infos) = jax.lax.scan(one_step, state, keys, unroll=10)


# In[15]:


thin_factor = 1
posterior_samples = []

for i in range(0, num_samples, thin_factor):
    sample_params = jtu.tree_map(lambda samples, i=i: samples[i], states.position)
    model = eqx.combine(sample_params, static)
    model = paramax.unwrap(model)
    latent_dist = model.predict(xtest, train_data=D)
    predictive_dist = model.likelihood(latent_dist)
    posterior_samples.append(predictive_dist.sample(key=key, sample_shape=(10,)))

posterior_samples = jnp.vstack(posterior_samples)
lower_ci, upper_ci = jnp.percentile(posterior_samples, jnp.array([2.5, 97.5]), axis=0)
expected_val = jnp.mean(posterior_samples, axis=0)


# In[16]:


fig, ax = plt.subplots()
ax.plot(t, loc, "o", markersize=5, color=cols[1], label="Observations", zorder=2, alpha=0.7)
ax.plot(xtest, expected_val, linewidth=2, color=cols[0], label="Predicted mean", zorder=1)
ax.fill_between(
    xtest.flatten(),
    lower_ci.flatten(),
    upper_ci.flatten(),
    alpha=0.2,
    color=cols[0],
    label="95% CI",
)


# In[17]:


predictive_dist.concentration


# In[18]:


plt.plot(xtest,predictive_dist.loc)


# In[ ]:




