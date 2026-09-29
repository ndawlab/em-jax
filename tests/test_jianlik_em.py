"""Simulate-and-fit smoke test for jianlik through the full pipeline
(simjian -> EMModel -> em_fit -> emerrors), which
tests/test_jianlik.py doesn't cover (that one only checks the per-subject
NLL against Julia, never a full fit). Not a Julia-exactness check -- just
confirms the whole pipeline recovers sane parameter estimates from data
simjian itself generated, with the expected significant/null pattern on
the covariate."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import pyem as em


def test_em_fit_recovers_jianlik_params():
    rng = np.random.default_rng(2026)
    NS, NT = 60, 150

    cov = rng.standard_normal(NS)
    cov -= cov.mean()

    true_beta = 1.0 + 0.5 * rng.standard_normal(NS) + cov  # real covariate effect
    true_nu = 0.0 + 1.0 * rng.standard_normal(NS)  # null covariate effect

    rows = []
    for i in range(NS):
        c, r = em.simjian([true_beta[i], true_nu[i]], NT, rng=rng)
        for t in range(NT):
            rows.append((i + 1, int(c[t]), int(r[t])))
    data = pd.DataFrame(rows, columns=["sub", "c", "r"])

    subs = list(range(1, NS + 1))
    X = jnp.stack([jnp.ones(NS), jnp.array(cov)], axis=1)

    model = em.EMModel(data, X, subs=subs, likfun=em.jianlik_nll,
                        nparam=2, reg_names=["Intercept", "Cov"], param_names=["Beta", "Nu"])
    startbetas = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    startsigma = jnp.array([1.0, 1.0])
    fit = em.em_fit(model, startbetas, startsigma, emtol=1e-3, quiet=0)

    betas = np.array(fit.betas)
    assert np.all(np.isfinite(betas))
    # intercepts should land near the true group means
    assert betas[0, 0] == pytest.approx(1.0, abs=0.3)
    assert betas[0, 1] == pytest.approx(0.0, abs=0.3)
    # real covariate effect on beta (coefficient 1.0 in true_beta's formula
    # above) should be recovered with the right sign/magnitude
    assert betas[1, 0] == pytest.approx(1.0, abs=0.4)

    errs = em.emerrors(fit)
    pvalues = np.array(errs.pvalues)
    # Cov:Beta (real effect) significant, Cov:Nu (null) not
    assert pvalues[2] < 0.01
    assert pvalues[3] > 0.05
