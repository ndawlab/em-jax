"""Simulate-and-fit smoke test for seqlik through the full pipeline
(simseq -> EMModel -> em_fit -> emerrors), which
tests/test_seqlik.py and test_seqlik_missing.py don't cover (those only
check the per-subject NLL against Julia, never a full fit). Not a
Julia-exactness check -- just confirms the whole pipeline recovers sane
parameter estimates from data simseq itself generated, with the expected
significant/null pattern on the covariate."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import pyem as em


def test_em_fit_recovers_seqlik_params():
    rng = np.random.default_rng(2026)
    NS, NT = 50, 150

    cov = rng.standard_normal(NS)
    cov -= cov.mean()

    # beta1m has a real covariate effect (coefficient 1.0); everything else null
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

    subs = list(range(1, NS + 1))
    X = jnp.stack([jnp.ones(NS), jnp.array(cov)], axis=1)

    model = em.EMModel(data, X, cols=("ch1", "ch2", "mn", "st"), subs=subs,
                        likfun=em.seqlik_nll, nparam=6,
                        reg_names=["Intercept", "Cov"],
                        param_names=["beta1m", "beta1t0", "beta1t1", "beta2", "lr", "ps"])
    startbetas = jnp.zeros((2, 6)).at[0, :].set(jnp.array([1.0, 0.5, 0.8, 1.0, 0.0, 0.2]))
    startsigma = jnp.ones(6)
    fit = em.em_fit(model, startbetas, startsigma, emtol=1e-3, maxiter=100, quiet=0)

    betas = np.array(fit.betas)
    assert np.all(np.isfinite(betas))
    # beta1m intercept and its real covariate effect should land in the right ballpark
    assert betas[0, 0] == pytest.approx(1.5, abs=0.4)
    assert betas[1, 0] == pytest.approx(1.0, abs=0.5)

    errs = em.emerrors(fit)
    pvalues = np.array(errs.pvalues)
    # Cov:beta1m (real effect) significant; Cov:beta1t0 (null) not
    assert pvalues[0] < 0.05  # Intercept:beta1m (obviously nonzero)
    assert pvalues[6] < 0.01  # Cov:beta1m (real effect)
    assert pvalues[7] > 0.05  # Cov:beta1t0 (null)
