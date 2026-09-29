import jax.numpy as jnp
import numpy as np
from scipy.stats import t as tdist

from .structs import EMErrors


def _is_diag(sigma):
    return sigma.ndim == 1


def _ncov_params(sigma):
    if _is_diag(sigma):
        return sigma.shape[0]
    nparam = sigma.shape[0]
    return nparam * (nparam + 1) // 2


def informationmatrixsigma_diag(sigma_diag, dof):
    d = 1.0 / sigma_diag
    return jnp.diag(dof / 2 * d ** 2)


def informationmatrixsigma_full(sigma, dof):
    """Sub-block of the complete-data information matrix for the unique
    (upper-triangular) sigma parameters, full/dense-covariance case.

    Args:
        sigma: (nparam, nparam) group covariance matrix.
        dof: effective sample size for sigma, nsub - nreg (see mstep's
            REML-equivalent correction), not nsub itself.

    Returns:
        (ncov, ncov) information sub-matrix, ncov = nparam*(nparam+1)/2.
    """
    nparam = sigma.shape[0]
    A = jnp.linalg.inv(sigma)
    pairs = [(i, j) for i in range(nparam) for j in range(i, nparam)]

    rows = []
    for (i, j) in pairs:
        row = []
        for (k, l) in pairs:
            if i == j and k == l:
                val = A[i, k] ** 2
            elif i != j and k == l:
                val = 2.0 * A[i, k] * A[j, k]
            elif i == j and k != l:
                val = 2.0 * A[i, k] * A[i, l]
            else:
                val = 2.0 * (A[i, k] * A[j, l] + A[i, l] * A[j, k])
            row.append((dof / 2) * val)
        rows.append(jnp.stack(row))
    return jnp.stack(rows)


def informationmatrixsigma(sigma, dof):
    """Sub-block of the complete-data information matrix for the group
    covariance parameters, dispatching on covariance shape.

    Args:
        sigma: (nparam,) vector (diagonal group covariance) or
            (nparam, nparam) matrix (full/dense).
        dof: effective sample size for sigma, nsub - nreg.

    Returns:
        Information sub-matrix for sigma's free parameters.
    """
    if _is_diag(sigma):
        return informationmatrixsigma_diag(sigma, dof)
    return informationmatrixsigma_full(sigma, dof)


def missing_information_diag(X, betas, sigma_diag, x, h, nreg, nparam):
    """Missing-information term (Louis, 1982) for the group-level
    parameters, diagonal-covariance case.

    Args:
        X: (nsub, nreg) design matrix.
        betas: (nreg, nparam) group-level regression coefficients.
        sigma_diag: (nparam,) diagonal group covariance.
        x: (nsub, nparam) per-subject parameter estimates.
        h: (nsub, nparam, nparam) per-subject Laplace covariances.
        nreg: number of regressors.
        nparam: number of subject-level parameters.

    Returns:
        (ntheta, ntheta) missing-information matrix, ntheta = nreg*nparam
        + nparam.
    """
    nsub = X.shape[0]
    d = 1.0 / sigma_diag
    mu = X @ betas
    residual = x - mu  # (nsub, nparam)

    ntheta = nreg * nparam + nparam
    J = jnp.zeros((nsub, ntheta, nparam))

    row = 0
    for r in range(nreg):
        for p in range(nparam):
            J = J.at[:, row, p].set(X[:, r] * d[p])
            row += 1
    for j in range(nparam):
        J = J.at[:, row, j].set(d[j] ** 2 * residual[:, j])
        row += 1

    return jnp.einsum("nab,nbc,ndc->ad", J, h, J)


def missing_information_full(X, betas, sigma, x, h, nreg, nparam):
    """Missing-information term (Louis, 1982) for the group-level
    parameters, full/dense-covariance case.

    Args:
        X: (nsub, nreg) design matrix.
        betas: (nreg, nparam) group-level regression coefficients.
        sigma: (nparam, nparam) group covariance matrix.
        x: (nsub, nparam) per-subject parameter estimates.
        h: (nsub, nparam, nparam) per-subject Laplace covariances.
        nreg: number of regressors.
        nparam: number of subject-level parameters.

    Returns:
        (ntheta, ntheta) missing-information matrix, ntheta = nreg*nparam
        + nparam*(nparam+1)/2.

    Notes:
        The covariance block unrolls the unique (upper-triangular) sigma
        parameters, each mapped back to its (symmetric) position(s) in the
        nparam x nparam matrix.
    """
    nsub = X.shape[0]
    sigma_inv = jnp.linalg.inv(sigma)
    mu = X @ betas
    residual = x - mu  # (nsub, nparam)

    ncov_params = nparam * (nparam + 1) // 2
    ntheta = nreg * nparam + ncov_params
    J = jnp.zeros((nsub, ntheta, nparam))

    row = 0
    for r in range(nreg):
        for p in range(nparam):
            J = J.at[:, row, :].set(X[:, r:r + 1] * sigma_inv[:, p][None, :])
            row += 1

    for r_cov in range(nparam):
        for c_cov in range(r_cov, nparam):
            E = jnp.zeros((nparam, nparam))
            E = E.at[r_cov, c_cov].set(1.0)
            if r_cov != c_cov:
                E = E.at[c_cov, r_cov].set(1.0)
            M = sigma_inv @ E @ sigma_inv
            J = J.at[:, row, :].set(residual @ M)
            row += 1

    return jnp.einsum("nab,nbc,ndc->ad", J, h, J)


def missing_information(X, betas, sigma, x, h, nreg, nparam):
    """Missing-information term (Louis, 1982) for the group-level
    parameters, dispatching on covariance shape.

    Args:
        X: (nsub, nreg) design matrix.
        betas: (nreg, nparam) group-level regression coefficients.
        sigma: (nparam,) vector (diagonal) or (nparam, nparam) matrix
            (full/dense) group covariance.
        x: (nsub, nparam) per-subject parameter estimates.
        h: (nsub, nparam, nparam) per-subject Laplace covariances.
        nreg: number of regressors.
        nparam: number of subject-level parameters.

    Returns:
        Missing-information matrix for the group-level parameters.
    """
    if _is_diag(sigma):
        return missing_information_diag(X, betas, sigma, x, h, nreg, nparam)
    return missing_information_full(X, betas, sigma, x, h, nreg, nparam)


# ---- public, EMFit-taking API ----

def groupinformation(fit):
    """Full observed (Louis-corrected) information matrix for all
    group-level parameters.

    Args:
        fit: an EMFit as returned by em_fit.

    Returns:
        (ntheta, ntheta) information matrix, ordered betas then sigma
        (matching packparams order).
    """
    X, betas, sigma, x, h = fit.model.X, fit.betas, fit.sigma, fit.x, fit.h
    nreg, nparam = betas.shape
    nsub = X.shape[0]
    dof = nsub - nreg
    nbetas = nreg * nparam
    sigma_full = jnp.diag(sigma) if _is_diag(sigma) else sigma

    h1_beta = jnp.linalg.inv(jnp.kron(jnp.linalg.inv(X.T @ X), sigma_full))
    h1_sigma = informationmatrixsigma(sigma, dof)
    ncov = h1_sigma.shape[0]
    ntheta = nbetas + ncov
    h1 = jnp.zeros((ntheta, ntheta))
    h1 = h1.at[:nbetas, :nbetas].set(h1_beta)
    h1 = h1.at[nbetas:, nbetas:].set(h1_sigma)

    h2 = missing_information(X, betas, sigma, x, h, nreg, nparam)
    return h1 - h2


def lml(fit):
    """Laplace approximation to the negative integrated log-likelihood of
    the dataset, i.e. -log p(data | betas, sigma).

    Args:
        fit: an EMFit as returned by em_fit.

    Returns:
        Scalar. Lower is better, matching the same convention as fit.l;
        ibic/iaic add their complexity penalties directly to this value.
    """
    nparam = fit.betas.shape[1]
    logdet_h = jnp.linalg.slogdet(fit.h)[1]  # h: (nsub, nparam, nparam), vmapped slogdet
    return -nparam / 2 * jnp.log(2 * jnp.pi) * fit.model.nsub + jnp.sum(fit.l) - jnp.sum(logdet_h) / 2


def ilaplace(fit):
    """Laplace approximation to the (negative) integrated log-likelihood,
    correcting for the group-level parameters' own estimation uncertainty.

    Args:
        fit: an EMFit as returned by em_fit.

    Returns:
        Scalar model comparison metric; lower is better.

    Notes:
        Uses the observed information matrix 
        rather than ibic's generic k/2*log(ndata) penalty.
    """
    nreg, nparam = fit.betas.shape
    k = nreg * nparam + _ncov_params(fit.sigma)
    _, logdet_info = jnp.linalg.slogdet(groupinformation(fit))
    return lml(fit) + logdet_info / 2 - k / 2 * jnp.log(2 * jnp.pi)


def ibic(fit, ndata=None):
    """Integrated BIC: lml(fit) + k/2*log(ndata), k = number of group-level
    parameters.

    Args:
        fit: an EMFit as returned by em_fit.
        ndata: sample size for the BIC penalty. Defaults to fit.model.nsub,
            since that's what actually governs the growth of the
            group-level Fisher information in this model, not
            nsub*ntrials. Pass ndata explicitly (e.g. nsub*ntrials) for
            Huys et al. 2011's original iBIC convention instead.

    Returns:
        Scalar model comparison metric; lower is better.
    """
    nreg, nparam = fit.betas.shape
    if ndata is None:
        ndata = fit.model.nsub
    k = nreg * nparam + _ncov_params(fit.sigma)
    return lml(fit) + k / 2 * jnp.log(ndata)


def iaic(fit):
    """Integrated AIC: lml(fit) + k, k = number of group-level parameters.

    Args:
        fit: an EMFit as returned by em_fit.

    Returns:
        Scalar model comparison metric; lower is better.
    """
    nreg, nparam = fit.betas.shape
    k = nreg * nparam + _ncov_params(fit.sigma)
    return lml(fit) + k


def emerrors(fit):
    """Standard errors and p-values for the group-level betas, via the
    missing-information-corrected group-level covariance.

    Args:
        fit: an EMFit as returned by em_fit.

    Returns:
        An EMErrors structure.
    """
    nreg, nparam = fit.betas.shape
    nbetas = nreg * nparam
    covmtx = jnp.linalg.inv(groupinformation(fit))
    covmtx_beta = covmtx[:nbetas, :nbetas]
    ses = jnp.sqrt(jnp.diag(covmtx_beta))
    dof = fit.model.nsub - nreg
    betas_flat = fit.betas.flatten()  # row-major == reg-major vec(betas')
    tvals = np.abs(np.array(betas_flat)) / np.array(ses)
    pvalues = 2 * tdist.sf(tvals, dof)
    return EMErrors(ses, pvalues, covmtx_beta, fit)
