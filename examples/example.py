"""pyem quickstart: simulate data from a Q-learning model, fit the
hierarchical model to recover it, and inspect the results. Also shows the
same walkthrough for a likelihood whose data isn't just (choices,
rewards) -- the two-step task model, seqlik -- which needs four per-trial
arrays instead of two.
"""
import os
# this allows threading over CPU cores (change number as appropriate)
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

import jax
jax.config.update("jax_enable_x64", True)  # (JAX defaults to float32)
import jax.numpy as jnp
import numpy as np
import pandas as pd

import pyem as em

# ---- simulate some Q-learning data ----
rng = np.random.default_rng(1234)
NS, NT = 100, 200

cov = rng.standard_normal(NS)
cov -= cov.mean()  # simulated between-subject covariate, e.g. age

beta_temp = 1.0 + 0.5 * rng.standard_normal(NS) + cov  # softmax temp: mean 1, effect of cov
beta_lr = 0.0 + 1.0 * rng.standard_normal(NS)  # Phi^-1(learning rate): mean 0, no cov effect

rows = []
for i in range(NS):
    c, r = em.simq([beta_temp[i], beta_lr[i]], NT, rng=rng)
    for t in range(NT):
        rows.append((i + 1, int(c[t]), int(r[t])))
data = pd.DataFrame(rows, columns=["sub", "c", "r"])

# subs fixes the subject order shared by X (below) and the padded per-trial
# data EMModel builds internally from data -- here it's just the subject
# ids in the same order the covariate arrays above were built in.
subs = list(range(1, NS + 1))

# ---- set up the model fit ----
# design matrix: one row per subject (in subs order), one column per predictor (here: intercept + cov)
X = jnp.stack([jnp.ones(NS), jnp.array(cov)], axis=1)

# build the model structure (names are optional)
model = em.EMModel(data, X, subs=subs, likfun=em.qlik_nll, nparam=2,
                    reg_names=["Intercept", "Cov"], param_names=["Temp", "LR"])

# start points
startbetas = jnp.array([[1.0, 0.0], [0.0, 0.0]])  # (nreg, nparam)
startsigma = jnp.array([5.0, 1.0])  # one starting variance per model parameter

# options
full = False  # fit diagonal covariance matrix (full covar: full = True)
skewcorrect = True  # optional 3rd-order correction for skewness (slower!)
quiet = 10  # print update every N iterations, or 0 not at all

# here is how the same setup would look if you just fit the model itself without the covariate
# (ie just an intercept)

# X = jnp.ones(NS)
#
# model = em.EMModel(data, X, subs=subs, likfun=em.qlik_nll, nparam=2,
#                     reg_names=["Intercept"], param_names=["Temp", "LR"])
#
# startbetas = jnp.array([1.0, 0.0])
# startsigma = jnp.array([5.0, 1.0])

# ---- fit the model ----
fit = em.em_fit(model, startbetas, startsigma, emtol=1e-3, full=full, quiet=quiet)
print(fit)

# ---- standard errors / p-values on the betas ----
errs = em.emerrors(fit)
print(errs)

# Cov:Temp should come out significant; Cov:LR should not (it's null).

# Optional 3rd-order correction for skewness (slower!)
# In practice, skewness can be a problem for parameters like LR due to boundaries
# and can show up as inflated Type I error rates for covariates on them
# such as cov:LR here

fit_skewcorrect = em.em_fit(model, startbetas, startsigma, emtol=1e-3, full=full, quiet=quiet, skewcorrect=True)
print(fit_skewcorrect)

errs_skewcorrect = em.emerrors(fit_skewcorrect)
print(errs_skewcorrect)  # cov on LR is indeed reduced!

# ---- model comparison / scoring ----

# all these metrics are aggregate measures of model fit, based on the
# sum over subjects of their log likelhoods, using a laplace approximation to
# the marginal likelihoods of the *subject-level* parameters
# (to correct for overfitting)

# this is the raw sum:

print("lml:     ", float(em.lml(fit)))  # aggregate negative log marginal likelihood

# these three additionally correct (using different methods) for the
# *group-level* parameters
# in the manner of Huys et al's (2010) integrated BIC

print("ilaplace:", float(em.ilaplace(fit)))
print("ibic:    ", float(em.ibic(fit)))
print("iaic:    ", float(em.iaic(fit)))

# although models can be compared using one of the aggregate scores above,
# it is generally preferred to compare per-subject (so that, if the relative
# fit varies from subject to subject, you take this into account to decide
# if a difference is significant).
#
# So we typically compute per-subject, per-model scores and compare using
# paired t test (or submit to something like SPM_BMS)
#
# However, per-subject scores from above fits are nonindependent due to
# shared group-level prior: to get a fair score we cross validate across
# subjects (of course, this is slow)

liks = em.loocv(fit)

# iaic tends to be a very good approximation of the sum
# though again the main idea here is breaking it down by subject
print(sum(liks))


# ---- same idea but for more complicated task / model (two-step task) ----
# ---- simulate some two-step task data ----

NS, NT = 100, 200

cov = rng.standard_normal(NS)
cov -= cov.mean()  # simulated between-subject covariate, e.g. age

# beta1m (the model-based weight on the first-stage decision) has a real
# covariate effect; every other parameter is null -- so we know what
# emerrors should find significant below.
true_beta1m = 1.5 + 0.3 * rng.standard_normal(NS) + cov
true_beta1t0 = 0.5 + 0.2 * rng.standard_normal(NS)
true_beta1t1 = 0.8 + 0.2 * rng.standard_normal(NS)
true_beta2 = 1.2 + 0.3 * rng.standard_normal(NS)
true_lr_raw = 0.0 + 1.0 * rng.standard_normal(NS)
true_ps = 0.2 + 0.3 * rng.standard_normal(NS)

rows = []
for i in range(NS):
    params = [true_beta1m[i], true_beta1t0[i], true_beta1t1[i], true_beta2[i],
              true_lr_raw[i], true_ps[i]]
    c1, s, c2, r = em.simseq(params, NT, rng=rng)
    for t in range(NT):
        rows.append((i + 1, int(c1[t]), int(c2[t]), int(r[t]), int(s[t])))
data = pd.DataFrame(rows, columns=["sub", "ch1", "ch2", "mn", "st"])

# ---- build the model ----
subs = list(range(1, NS + 1))
X = jnp.stack([jnp.ones(NS), jnp.array(cov)], axis=1)

# the key difference is you need to tell EMModel which columns to extract
# for the per-trial data (seqlik needs four, not two: first-stage
# choice, second-stage choice, reward, second-stage state).
model = em.EMModel(data, X, cols=("ch1", "ch2", "mn", "st"), subs=subs,
                    likfun=em.seqlik_nll, nparam=6,
                    reg_names=["Intercept", "Cov"],
                    param_names=["beta1m", "beta1t0", "beta1t1", "beta2", "lr", "ps"])

startbetas = jnp.zeros((2, 6)).at[0, :].set(jnp.array([1.0, 0.5, 0.8, 1.0, 0.0, 0.2]))
startsigma = jnp.ones(6)  # one starting variance per model parameter

# ---- fit it ----
fit = em.em_fit(model, startbetas, startsigma, emtol=1e-3, full=full, quiet=quiet)
print(fit)

# error bars
errs = em.emerrors(fit)
print(errs)
# Cov:beta1m should come out significant (the real effect); the other five
# Cov: rows should not (they're null).

# model selection metrics
print("\nlml:     ", float(em.lml(fit)))
print("ilaplace:", float(em.ilaplace(fit)))
