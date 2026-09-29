import jax.numpy as jnp
from tqdm.auto import tqdm
from .emloop import em_fit
from .estep import subject_laplace_fit
from .structs import EMModel


def heldoutsubject_laplace(mu, sigma, data, valid_mask, likfun, x_init, skewcorrect=False,
                            skewcorrect_maxcorr=1.0):
    """Laplace-approximated held-out score for one subject, given group
    parameters fit WITHOUT this subject.

    Args:
        mu: (nparam,) group-level mean for this subject (e.g. X[i] @
            betas from a leave-one-out fit).
        sigma: (nparam,) vector (diagonal) or (nparam, nparam) matrix
            (full) group covariance, matching whatever em_fit was run
            with.
        data: tuple of per-trial data arrays for this one subject.
        valid_mask: per-trial validity mask for this subject.
        likfun: the model's negative-log-likelihood function.
        x_init: (nparam,) initial parameter guess for the MAP fit.
        skewcorrect: whether to apply the skewness correction (see
            subject_laplace_fit).
        skewcorrect_maxcorr: maximum allowed correction size, passed
            through when skewcorrect=True.

    Returns:
        Scalar held-out score. Same nll-loss convention as lml/ibic/iaic:
        lower is better.

    Notes:
        Direct port of EM.jl's heldoutsubject_laplace.
    """
    nparam = mu.shape[0]
    sigma_full = jnp.diag(sigma) if sigma.ndim == 1 else sigma
    inv_sigma = jnp.linalg.inv(sigma_full)
    logdet_sigma = jnp.linalg.slogdet(sigma_full)[1]
    _x_hat, l, h, _hfb, _sfb = subject_laplace_fit(
        x_init, mu, inv_sigma, logdet_sigma, data, valid_mask, likfun,
        skewcorrect=skewcorrect, maxcorr=skewcorrect_maxcorr)
    H = jnp.linalg.inv(h)
    return -nparam / 2 * jnp.log(2 * jnp.pi) + l + jnp.linalg.slogdet(H)[1] / 2


def loocv(fit, emtol=1e-3, maxiter=100, skewcorrect=False, skewcorrect_maxcorr=1.0, progress=True):
    """Leave-one-subject-out predictive score for each subject: refit the
    group-level parameters on every other subject, then score the held-out
    subject's own data under those cross-validated group parameters via a
    Laplace approximation.

    Args:
        fit: an EMFit as returned by em_fit; its betas/sigma/x are used to
            warm-start each leave-one-out refit.
        emtol: convergence tolerance for each leave-one-out em_fit call.
        maxiter: maximum outer EM iterations for each leave-one-out
            em_fit call.
        skewcorrect: whether to apply the skewness correction in both the
            leave-one-out refits and the held-out subject's own
            evaluation. Not inferred from fit (nothing records how it was
            produced), so pass the same value used to produce fit for
            consistency between the two.
        skewcorrect_maxcorr: maximum allowed correction size, passed
            through when skewcorrect=True.
        progress: if True, shows a live progress bar (via tqdm.auto) over
            the nsub leave-one-out refits.

    Returns:
        (nsub,) array of held-out scores, one per subject. Same nll-loss
        convention as lml/ibic/iaic (lower is better); sum(loocv(...)) is
        directly comparable to iaic(fit).

    Notes:
        Direct port of EM.jl's loocv. Refits the model nsub times, so it's
        inherently expensive. Each internal em_fit call runs with quiet=0
        (fully silent), so periodic per-refit status blocks don't
        interleave with the progress bar.
    """
    model = fit.model
    X, data, valid_mask, likfun = model.X, model.data, model.valid_mask, model.likfun
    nsub = X.shape[0]
    full = fit.sigma.ndim == 2  # fit.sigma's own shape unambiguously says which fit produced it
    liks = []
    idx_all = jnp.arange(nsub)
    for i in tqdm(range(nsub), desc="loocv", unit="subject", disable=not progress):
        mask = idx_all != i
        loo_data = tuple(a[mask] for a in data)
        loo_model = EMModel.from_arrays(X[mask], *loo_data, valid_mask=valid_mask[mask], likfun=likfun,
                                         nparam=model.nparam, reg_names=model.reg_names,
                                         param_names=model.param_names)
        loo_fit = em_fit(loo_model, fit.betas, fit.sigma, emtol=emtol, maxiter=maxiter,
                          startx=fit.x[mask], full=full, skewcorrect=skewcorrect,
                          skewcorrect_maxcorr=skewcorrect_maxcorr, quiet=0)
        newmu = loo_fit.betas.T @ X[i]
        held_out_data = tuple(a[i] for a in data)
        lik = heldoutsubject_laplace(newmu, loo_fit.sigma, held_out_data, valid_mask[i], likfun,
                                      x_init=fit.x[i], skewcorrect=skewcorrect,
                                      skewcorrect_maxcorr=skewcorrect_maxcorr)
        liks.append(lik)
    return jnp.array(liks)
