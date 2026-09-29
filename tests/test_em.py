"""End-to-end validation of em_fit + emerrors + ilaplace against Julia's
em() + emerrors() + ilaplace() (tests/fixtures/reference_fit.txt, generated
by scripts/gen_reference_full.jl -- 40 subjects, ragged trial counts,
covariate-driven prior mean)."""
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from pyem.likelihoods import qlik_nll
from pyem.structs import EMModel
from pyem.emloop import em_fit
from pyem.errors import emerrors, ilaplace, ibic, iaic, lml

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def reference():
    with open(FIXTURES / "reference_fit.txt") as f:
        ref = {}
        for line in f:
            parts = line.strip().split(",")
            key = parts[0]
            if key == "x":
                continue
            ref[key] = [float(v) for v in parts[1:]]
    return ref


@pytest.fixture(scope="module")
def fitted():
    df = pd.read_csv(FIXTURES / "reference_data_full.csv")
    covdf = pd.read_csv(FIXTURES / "reference_cov.csv")
    subs = sorted(df["sub"].unique())
    cov = jnp.array(covdf.sort_values("sub")["cov"].to_numpy())
    nsub = len(subs)
    X = jnp.stack([jnp.ones(nsub), cov], axis=1)

    model = EMModel(df, X, subs=subs, likfun=qlik_nll, nparam=2,
                     reg_names=["Intercept", "Cov"], param_names=["Temp", "LR"])
    startbetas = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    startsigma = jnp.array([1.0, 1.0])
    fit = em_fit(model, startbetas, startsigma, emtol=1e-4, maxiter=200, quiet=0)
    return fit


def test_em_fit_matches_julia(fitted, reference):
    fit = fitted
    np.testing.assert_allclose(np.array(fit.betas.flatten()), reference["betas"], atol=1e-4)
    np.testing.assert_allclose(np.array(fit.sigma), reference["sigma"], atol=1e-4)


def test_emerrors_matches_julia(fitted, reference):
    fit = fitted
    errs = emerrors(fit)
    np.testing.assert_allclose(np.array(errs.ses), reference["ses"], atol=1e-4)
    np.testing.assert_allclose(np.array(errs.pvalues), reference["pvalues"], atol=1e-4)


def test_emerrors_repr_is_labeled_table(fitted):
    # the whole point of EMErrors's pretty-printer: reg_names/param_names
    # actually show up, not just raw arrays
    text = repr(emerrors(fitted))
    assert "Intercept" in text and "Cov" in text
    assert "Temp" in text and "LR" in text
    assert "Estimate" in text and "Std.Error" in text and "p-value" in text


def test_ilaplace_matches_julia(fitted, reference):
    fit = fitted
    il = ilaplace(fit)
    assert float(il) == pytest.approx(reference["ilaplace"][0], abs=1e-2)


def test_lml_matches_julia(fitted, reference):
    fit = fitted
    val = lml(fit)
    assert float(val) == pytest.approx(reference["lml"][0], abs=1e-2)


def test_ibic_iaic_analytic(fitted, reference):
    # ibic/iaic are simple closed-form corrections to lml (no new Julia
    # reference needed): ibic = lml + k/2*log(ndata), iaic = lml + k,
    # k = nreg*nparam + nparam. Check both conventions of ndata.
    fit = fitted
    nsub, nreg, nparam = fit.model.nsub, fit.model.nreg, fit.model.nparam
    k = nreg * nparam + nparam
    lml_val = reference["lml"][0]

    bic_default = ibic(fit)
    assert float(bic_default) == pytest.approx(lml_val + k / 2 * np.log(nsub), abs=1e-2)

    ndata = nsub * 150  # Huys-style convention, arbitrary ntrials for this check
    bic_explicit = ibic(fit, ndata=ndata)
    assert float(bic_explicit) == pytest.approx(lml_val + k / 2 * np.log(ndata), abs=1e-2)

    aic = iaic(fit)
    assert float(aic) == pytest.approx(lml_val + k, abs=1e-2)
