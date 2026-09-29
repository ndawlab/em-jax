import jax.numpy as jnp


def mstep(X, x, h, nreg, full=False):
    """Closed-form M-step: updates the group-level betas and sigma from
    the current per-subject fits.

    Args:
        X: (nsub, nreg) design matrix.
        x: (nsub, nparam) per-subject parameter estimates.
        h: (nsub, nparam, nparam) per-subject Laplace covariances.
        nreg: number of regressors (X.shape[1]).
        full: if True, sigma is returned as a full/dense (nparam, nparam)
            matrix; if False (default), as a diagonal (nparam,) vector.

    Returns:
        (betas, sigma): betas is (nreg, nparam).

    Notes:
        Direct port of EM.jl's mstep!.
    """
    nsub = X.shape[0]
    XtX_inv = jnp.linalg.inv(X.T @ X)
    betas = XtX_inv @ X.T @ x  # (nreg, nparam)

    proj = jnp.eye(nsub) - X @ XtX_inv @ X.T
    newsigma = (x.T @ proj @ x + jnp.sum(h, axis=0)) / (nsub - nreg)
    if full:
        return betas, newsigma
    return betas, jnp.diag(newsigma)
