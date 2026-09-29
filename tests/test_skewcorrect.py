"""Validates the skewness correction and the positive-definiteness safety
net (safe_laplace_hessian / subject_laplace_fit)."""
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from pyem.data import _pad_by_subject
from pyem.likelihoods import qlik_nll
from pyem.structs import EMModel
from pyem.emloop import em_fit
from pyem.errors import emerrors, ilaplace, lml
from pyem.estep import safe_laplace_hessian, subject_laplace_fit

FIXTURES = Path(__file__).parent / "fixtures"


def test_safe_laplace_hessian_falls_back_on_saddle():
    # x[1]^2 - x[2]^2: Hessian is diag(2, -2) everywhere, never PD.
    fitfun = lambda x: x[0] ** 2 - x[1] ** 2
    xhat = jnp.array([0.3, -0.7])
    inv_sigma = jnp.diag(jnp.array([2.0, 3.0]))
    H, fellback = safe_laplace_hessian(fitfun, xhat, inv_sigma)
    assert bool(fellback)
    np.testing.assert_allclose(np.array(H), np.array(inv_sigma))


def test_safe_laplace_hessian_keeps_pd_hessian():
    fitfun = lambda x: x[0] ** 2 + 2 * x[1] ** 2  # Hessian diag(2, 4), PD
    xhat = jnp.array([0.1, -0.2])
    inv_sigma = jnp.eye(2)
    H, fellback = safe_laplace_hessian(fitfun, xhat, inv_sigma)
    assert not bool(fellback)
    np.testing.assert_allclose(np.array(H), np.diag([2.0, 4.0]), atol=1e-8)


@pytest.fixture(scope="module")
def data_and_prior():
    df = pd.read_csv(FIXTURES / "reference_data.csv")
    subs = sorted(df["sub"].unique())
    padded, valid_mask = _pad_by_subject(df, subs)
    betas = jnp.array([0.8, 0.1])
    sigma = jnp.diag(jnp.array([2.0, 1.5]))
    inv_sigma = jnp.linalg.inv(sigma)
    logdet_sigma = jnp.linalg.slogdet(sigma)[1]
    return padded, valid_mask, betas, inv_sigma, logdet_sigma


def test_maxcorr_zero_matches_no_skewcorrect(data_and_prior):
    # dxsize > 0 always (unless the correction is exactly zero), so
    # maxcorr=0.0 should always discard the correction -- a deterministic
    # regression check on the fallback path itself.
    padded, valid_mask, betas, inv_sigma, logdet_sigma = data_and_prior
    for i in range(5):
        data = (padded["c"][i], padded["r"][i])
        x_plain, l_plain, h_plain, hfb_plain, sfb_plain = subject_laplace_fit(
            betas, betas, inv_sigma, logdet_sigma, data, valid_mask[i], qlik_nll, skewcorrect=False)
        x_zero, l_zero, h_zero, hfb_zero, sfb_zero = subject_laplace_fit(
            betas, betas, inv_sigma, logdet_sigma, data, valid_mask[i], qlik_nll,
            skewcorrect=True, maxcorr=0.0)
        np.testing.assert_allclose(np.array(x_plain), np.array(x_zero), atol=1e-8)
        np.testing.assert_allclose(np.array(h_plain), np.array(h_zero), atol=1e-8)
        assert float(l_plain) == pytest.approx(float(l_zero), abs=1e-8)
        assert bool(sfb_zero)  # correction was attempted and (deterministically) discarded


def test_skewcorrect_gives_valid_pd_results(data_and_prior):
    padded, valid_mask, betas, inv_sigma, logdet_sigma = data_and_prior
    for i in range(5):
        data = (padded["c"][i], padded["r"][i])
        x, l, h, hfb, sfb = subject_laplace_fit(
            betas, betas, inv_sigma, logdet_sigma, data, valid_mask[i], qlik_nll,
            skewcorrect=True, maxcorr=1.0)
        assert np.all(np.isfinite(np.array(x)))
        assert np.isfinite(float(l))
        eigvals = np.linalg.eigvalsh(np.array(h))
        assert np.all(eigvals > 0)  # h is a covariance -- must stay PD


@pytest.fixture(scope="module")
def reference_skewcorrect():
    with open(FIXTURES / "reference_fit_skewcorrect.txt") as f:
        ref = {}
        for line in f:
            parts = line.strip().split(",")
            ref[parts[0]] = [float(v) for v in parts[1:]]
    return ref


@pytest.fixture(scope="module")
def fitted_skewcorrect():
    df = pd.read_csv(FIXTURES / "reference_data_full.csv")
    covdf = pd.read_csv(FIXTURES / "reference_cov.csv")
    subs = sorted(df["sub"].unique())
    cov = jnp.array(covdf.sort_values("sub")["cov"].to_numpy())
    nsub = len(subs)
    X = jnp.stack([jnp.ones(nsub), cov], axis=1)

    model = EMModel(df, X, subs=subs, likfun=qlik_nll, nparam=2)
    startbetas = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    startsigma = jnp.array([1.0, 1.0])
    fit = em_fit(model, startbetas, startsigma, emtol=1e-4, maxiter=200,
                 skewcorrect=True, skewcorrect_maxcorr=1.0, quiet=0)
    return fit


def test_em_fit_skewcorrect_matches_julia(fitted_skewcorrect, reference_skewcorrect):
    fit = fitted_skewcorrect
    ref = reference_skewcorrect
    np.testing.assert_allclose(np.array(fit.betas.flatten()), ref["betas"], atol=1e-3)
    np.testing.assert_allclose(np.array(fit.sigma), ref["sigma"], atol=1e-3)

    errs = emerrors(fit)
    np.testing.assert_allclose(np.array(errs.ses), ref["ses"], atol=1e-3)
    np.testing.assert_allclose(np.array(errs.pvalues), ref["pvalues"], atol=1e-3)

    il = ilaplace(fit)
    assert float(il) == pytest.approx(ref["ilaplace"][0], abs=0.5)


def test_em_fit_skewcorrect_differs_from_plain(fitted_skewcorrect):
    # sanity check that skewcorrect is actually engaging on this dataset,
    # not silently discarding every correction
    fit = fitted_skewcorrect
    padded_betas = np.array(fit.betas.flatten())
    plain_betas = np.array([0.8494824191109887, -0.369792882450246, 1.0405184249219466,
                             0.2082976872043787])  # reference_fit.txt, same data, skewcorrect=False
    assert not np.allclose(padded_betas, plain_betas, atol=1e-3)
