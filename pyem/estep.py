import jax
import jax.numpy as jnp
import optimistix as optx


def gaussianprior_nll(x, mu, inv_sigma, logdet_sigma, data, valid_mask, likfun):
    """Negative log-posterior for one subject: nll - logprior, under a
    Gaussian prior on the subject-level parameters.

    Args:
        x: (nparam,) candidate subject-level parameter vector.
        mu: (nparam,) prior mean.
        inv_sigma: (nparam, nparam) prior precision matrix.
        logdet_sigma: log-determinant of the prior covariance.
        data: tuple of however many per-trial arrays likfun needs (e.g.
            (choices, rewards) for qlik/jianlik, (ch1, ch2, mn, st) for
            seqlik), passed straight through as likfun(x, *data,
            valid_mask). Kept as a tuple rather than *args so vmap/pmap's
            in_axes can address it as one nested pytree argument, e.g.
            in_axes=(..., (0,) * len(data), ...).
        valid_mask: per-trial validity mask, passed through to likfun.
        likfun: the model's negative-log-likelihood function.

    Returns:
        Scalar negative log-posterior.

    Notes:
        Direct port of EM.jl's gaussianprior.
    """
    d = x.shape[0]
    diff = x - mu
    quad = diff @ inv_sigma @ diff
    logprior = -0.5 * d * jnp.log(2 * jnp.pi) - 0.5 * logdet_sigma - 0.5 * quad
    nll = likfun(x, *data, valid_mask=valid_mask)
    return nll - logprior


def safe_laplace_hessian(fitfun, xhat, inv_sigma):
    """Hessian of fitfun at xhat, guarded to always be positive-definite.

    Args:
        fitfun: scalar function of a subject's parameter vector (typically
            a closure over gaussianprior_nll).
        xhat: (nparam,) point to evaluate the Hessian at, usually the MAP
            estimate.
        inv_sigma: (nparam, nparam) prior precision matrix, used as the
            fallback Hessian.

    Returns:
        (H, hessian_fallback): H is Hessian(fitfun)(xhat) if that is
        positive-definite, else inv_sigma; hessian_fallback is a bool
        flagging whether the fallback was used.

    Notes:
        inv_sigma is always PD and is the mathematically correct limiting
        Hessian for a subject whose data contributes no trustworthy
        curvature. Since the Hessian of gaussianprior is always
        Hessian(nll) + inv_sigma, a non-PD result specifically means the
        *data* term's own curvature broke that positive floor -- typically
        because the optimizer hasn't actually reached a local minimum yet.
    """
    H = jax.hessian(fitfun)(xhat)
    is_pd = jnp.all(jnp.linalg.eigvalsh(H) > 0)
    H_safe = jnp.where(is_pd, H, inv_sigma)
    return H_safe, ~is_pd


def subject_laplace_fit(x_init, mu, inv_sigma, logdet_sigma, data, valid_mask, likfun,
                         max_steps=256, skewcorrect=False, maxcorr=1.0):
    """MAP estimate plus (optionally skewness-corrected) Laplace covariance
    for one subject.

    Args:
        x_init: (nparam,) initial parameter guess.
        mu: (nparam,) prior mean.
        inv_sigma: (nparam, nparam) prior precision matrix.
        logdet_sigma: log-determinant of the prior covariance.
        data: tuple of per-trial data arrays for this subject, passed to
            likfun (see gaussianprior_nll).
        valid_mask: per-trial validity mask.
        likfun: the model's negative-log-likelihood function.
        max_steps: maximum LBFGS iterations for the MAP optimization.
        skewcorrect: if True, apply the skewness correction described in
            Notes.
        maxcorr: maximum allowed correction size (in Laplace-SD units)
            before it is discarded; only used when skewcorrect=True.

    Returns:
        (x, l, h, hessian_fallback, skew_fallback): x is the (corrected)
        parameter estimate, l is fitfun's value there, h = inv(H) is
        always a genuine covariance (H is always PD, by construction of
        safe_laplace_hessian), and hessian_fallback/skew_fallback are bool
        flags for whether each fallback was used.

    Notes:
        Direct port of EM.jl's subject_laplace_fit, shared by em_fit's
        E-step and loocv's held-out evaluation.

        The skewness correction is a first-order perturbative expansion of
        the posterior mode/mean gap (via the third-derivative tensor of the
        log-posterior at the mode), so it can misbehave for subjects where
        it's least trustworthy. Two checks guard against that: if the
        correction is large relative to the Laplace spread itself
        (Mahalanobis norm under H, vs. maxcorr), or if the Hessian at the
        corrected point isn't positive-definite (it isn't guaranteed to be,
        since the corrected point need not be a local minimum), the
        correction is discarded and the plain Laplace mode/Hessian is used
        instead.
    """
    d = x_init.shape[0]

    def fitfun(x):
        return gaussianprior_nll(x, mu, inv_sigma, logdet_sigma, data, valid_mask, likfun)

    solver = optx.LBFGS(rtol=1e-8, atol=1e-8)
    sol = optx.minimise(lambda x, _: fitfun(x), solver, x_init, max_steps=max_steps, throw=False)
    xhat = sol.value
    l0 = fitfun(xhat)
    H, hessian_fallback = safe_laplace_hessian(fitfun, xhat, inv_sigma)

    if not skewcorrect:
        return xhat, l0, jnp.linalg.inv(H), hessian_fallback, jnp.array(False)

    Hinv = jnp.linalg.inv(H)
    hessfun = lambda y: jax.hessian(fitfun)(y)
    T3 = jax.jacobian(hessfun)(xhat)  # T3[k, kk, j] = d H[k, kk] / d x_j

    S = jnp.einsum('abj,ab->j', T3, Hinv)
    dx = -0.5 * (Hinv @ S)
    dxsize = jnp.sqrt(dx @ H @ dx)  # size of the correction in Laplace-SD units

    xcorr = xhat + dx
    Hcorr_raw = jax.hessian(fitfun)(xcorr)
    Hcorr = 0.5 * (Hcorr_raw + Hcorr_raw.T)
    Hcorr_is_pd = jnp.all(jnp.linalg.eigvalsh(Hcorr) > 0)

    discard = hessian_fallback | (dxsize > maxcorr) | (~Hcorr_is_pd)
    skew_fallback = (~hessian_fallback) & discard

    x_final = jnp.where(discard, xhat, xcorr)
    l_final = jnp.where(discard, l0, fitfun(xcorr))
    H_final = jnp.where(discard, H, Hcorr)
    return x_final, l_final, jnp.linalg.inv(H_final), hessian_fallback, skew_fallback


def fit_subject(x_init, mu, inv_sigma, logdet_sigma, data, valid_mask, likfun, max_steps=256):
    """MAP estimate plus Laplace covariance for one subject, with the
    positive-definiteness safety net (see safe_laplace_hessian) but no
    skewness correction.

    Args:
        x_init: (nparam,) initial parameter guess.
        mu: (nparam,) prior mean.
        inv_sigma: (nparam, nparam) prior precision matrix.
        logdet_sigma: log-determinant of the prior covariance.
        data: tuple of per-trial data arrays for this subject.
        valid_mask: per-trial validity mask.
        likfun: the model's negative-log-likelihood function.
        max_steps: maximum LBFGS iterations for the MAP optimization.

    Returns:
        (x, h, l): parameter estimate, its Laplace covariance, and
        fitfun's value there.

    Notes:
        Equivalent to subject_laplace_fit(..., skewcorrect=False), kept as
        the simple 3-return-value entry point used by em_fit/loocv when
        skewcorrect isn't requested.
    """
    x, l, h, _hessian_fallback, _skew_fallback = subject_laplace_fit(
        x_init, mu, inv_sigma, logdet_sigma, data, valid_mask, likfun, max_steps=max_steps)
    return x, h, l
