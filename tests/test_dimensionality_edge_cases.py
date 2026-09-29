"""Confidence check on dimensionality edge cases nothing else in the test
suite exercises: every existing full-pipeline test uses nparam=2 AND
nreg=2 (an intercept + one covariate). This checks nparam=1 (a genuinely
single-parameter model -- none of qlik/jianlik/seqlik have one, so a
minimal fixed-learning-rate variant of qlik is built here just for this),
nreg=1 (intercept-only design, no covariates), and both together. Not
Julia-exactness checks (no likelihood like this exists in EM.jl to compare
against) -- these confirm the group-level linear-algebra plumbing
(kron products, the informationmatrixsigma/missing_information unrolling,
mstep's diag extraction, EMErrors's table layout) doesn't silently break
when a dimension collapses to 1."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax.scipy.special import logsumexp

import pyem as em


@em.scan_likelihood
def qlik_fixed_lr_nll(params):
    """A genuinely single-parameter (nparam=1) likelihood: qlik with the
    learning rate fixed at 0.5 rather than fit -- only beta (params[0]) is
    a free subject-level parameter."""
    beta = params[0]
    lr = 0.5

    def step(Q, c, r):
        idx = c - 1
        ll = beta * Q[idx] - logsumexp(beta * Q)
        Q_new = Q.at[idx].set((1 - lr) * Q[idx] + r)
        return Q_new, ll

    return jnp.zeros(2), step


def _simulate_fixed_lr(true_beta, ntrials, rng):
    # simq needs a (beta, raw_lr) pair to simulate; raw_lr=0.0 -> lr=0.5,
    # matching qlik_fixed_lr_nll's hardcoded lr exactly.
    return em.simq([true_beta, 0.0], ntrials, rng=rng)


def _check_fit_runs_end_to_end(model, startbetas, startsigma, nreg, nparam, full=False):
    fit = em.em_fit(model, startbetas, startsigma, emtol=1e-3, full=full, quiet=0)
    assert fit.betas.shape == (nreg, nparam)
    assert np.all(np.isfinite(np.array(fit.betas)))
    assert np.all(np.isfinite(np.array(fit.sigma)))
    assert fit.sigma.ndim == (2 if full else 1)

    errs = em.emerrors(fit)
    nbetas = nreg * nparam
    assert errs.ses.shape == (nbetas,)
    assert errs.pvalues.shape == (nbetas,)
    assert np.all(np.isfinite(np.array(errs.ses)))
    assert np.all(np.isfinite(np.array(errs.pvalues)))
    repr(errs)  # pretty-printer shouldn't crash on these shapes either

    for scorefn in (em.lml, em.ilaplace, em.ibic, em.iaic):
        val = float(scorefn(fit))
        assert np.isfinite(val)

    small_model = em.EMModel.from_arrays(model.X[:8], *[a[:8] for a in model.data],
                                          valid_mask=model.valid_mask[:8],
                                          likfun=model.likfun, nparam=nparam)
    small_fit = em.em_fit(small_model, startbetas, startsigma, emtol=1e-3, full=full, quiet=0)
    liks = em.loocv(small_fit, progress=False)
    assert liks.shape == (8,)
    assert np.all(np.isfinite(np.array(liks)))

    return fit, errs


def test_nparam_one_with_covariate():
    rng = np.random.default_rng(7)
    NS, NT = 40, 150
    cov = rng.standard_normal(NS)
    cov -= cov.mean()
    true_beta = 1.0 + 0.4 * rng.standard_normal(NS) + cov  # real covariate effect

    rows = []
    for i in range(NS):
        c, r = _simulate_fixed_lr(true_beta[i], NT, rng)
        for t in range(NT):
            rows.append((i + 1, int(c[t]), int(r[t])))
    data = pd.DataFrame(rows, columns=["sub", "c", "r"])
    subs = list(range(1, NS + 1))
    X = jnp.stack([jnp.ones(NS), jnp.array(cov)], axis=1)  # nreg=2

    model = em.EMModel(data, X, subs=subs, likfun=qlik_fixed_lr_nll,
                        nparam=1, reg_names=["Intercept", "Cov"], param_names=["Beta"])
    startbetas = jnp.array([[1.0], [0.0]])  # (nreg=2, nparam=1)
    startsigma = jnp.array([1.0])  # (nparam=1,) diagonal -- and the only sigma shape when nparam=1

    fit, errs = _check_fit_runs_end_to_end(model, startbetas, startsigma, nreg=2, nparam=1)
    betas = np.array(fit.betas)
    assert betas[0, 0] == pytest.approx(1.0, abs=0.3)
    assert betas[1, 0] == pytest.approx(1.0, abs=0.4)
    assert np.array(errs.pvalues)[1] < 0.01  # Cov:Beta, the real effect

    # full (dense) covariance with nparam=1 is a (1, 1) matrix -- degenerate
    # but should be handled identically to the diagonal case, not crash --
    # and mathematically must give numerically IDENTICAL results, not just
    # both-run-without-crashing, since diag/full are the same 1x1 matrix.
    # Same plain-vector startsigma as the diagonal case above -- full=True
    # is what decides it's expanded to a (1, 1) starting matrix, not the
    # vector's own shape.
    fit_full, _ = _check_fit_runs_end_to_end(model, startbetas, startsigma, nreg=2, nparam=1, full=True)
    assert fit_full.sigma.shape == (1, 1)
    np.testing.assert_allclose(np.array(fit.betas), np.array(fit_full.betas), atol=1e-6)
    np.testing.assert_allclose(np.array(fit.sigma), np.array(fit_full.sigma).flatten(), atol=1e-6)
    assert float(em.ilaplace(fit)) == pytest.approx(float(em.ilaplace(fit_full)), abs=1e-4)


def test_nreg_one_intercept_only():
    rng = np.random.default_rng(11)
    NS, NT = 40, 150
    true_beta = 1.0 + 0.5 * rng.standard_normal(NS)
    true_lr_raw = 0.0 + 1.0 * rng.standard_normal(NS)

    rows = []
    for i in range(NS):
        c, r = em.simq([true_beta[i], true_lr_raw[i]], NT, rng=rng)
        for t in range(NT):
            rows.append((i + 1, int(c[t]), int(r[t])))
    data = pd.DataFrame(rows, columns=["sub", "c", "r"])
    subs = list(range(1, NS + 1))
    X = jnp.ones((NS, 1))  # nreg=1, no covariates

    model = em.EMModel(data, X, subs=subs, likfun=em.qlik_nll,
                        nparam=2, reg_names=["Intercept"], param_names=["Temp", "LR"])
    startbetas = jnp.array([[1.0, 0.0]])  # (nreg=1, nparam=2)
    startsigma = jnp.array([1.0, 1.0])

    fit, errs = _check_fit_runs_end_to_end(model, startbetas, startsigma, nreg=1, nparam=2)
    betas = np.array(fit.betas)
    assert betas[0, 0] == pytest.approx(1.0, abs=0.3)
    assert betas[0, 1] == pytest.approx(0.0, abs=0.3)

    fit_full, _ = _check_fit_runs_end_to_end(model, startbetas, startsigma, nreg=1, nparam=2, full=True)
    assert fit_full.sigma.shape == (2, 2)

    # X as a plain (nsub,) vector and startbetas as a plain (nparam,)
    # vector -- the shorthand for an intercept-only design -- must give
    # EXACTLY the same model/fit as the explicit (nsub, 1)/(1, nparam)
    # construction above, not just "also runs without crashing"
    vector_model = em.EMModel(data, jnp.ones(NS), subs=subs, likfun=em.qlik_nll, nparam=2,
                               reg_names=["Intercept"], param_names=["Temp", "LR"])
    assert vector_model.X.shape == (NS, 1)
    vector_fit = em.em_fit(vector_model, jnp.array([1.0, 0.0]), startsigma, emtol=1e-3, quiet=0)
    assert vector_fit.betas.shape == (1, 2)
    np.testing.assert_allclose(np.array(vector_fit.betas), np.array(fit.betas), atol=1e-8)
    np.testing.assert_allclose(np.array(vector_fit.sigma), np.array(fit.sigma), atol=1e-8)


def test_nreg_one_and_nparam_one_together():
    rng = np.random.default_rng(13)
    NS, NT = 40, 150
    true_beta = 1.0 + 0.4 * rng.standard_normal(NS)

    rows = []
    for i in range(NS):
        c, r = _simulate_fixed_lr(true_beta[i], NT, rng)
        for t in range(NT):
            rows.append((i + 1, int(c[t]), int(r[t])))
    data = pd.DataFrame(rows, columns=["sub", "c", "r"])
    subs = list(range(1, NS + 1))
    X = jnp.ones((NS, 1))  # nreg=1

    model = em.EMModel(data, X, subs=subs, likfun=qlik_fixed_lr_nll,
                        nparam=1, reg_names=["Intercept"], param_names=["Beta"])
    startbetas = jnp.array([[1.0]])  # (1, 1)
    startsigma = jnp.array([1.0])  # (1,)

    fit, errs = _check_fit_runs_end_to_end(model, startbetas, startsigma, nreg=1, nparam=1)
    betas = np.array(fit.betas)
    assert betas[0, 0] == pytest.approx(1.0, abs=0.3)
    assert errs.covmtx.shape == (1, 1)
