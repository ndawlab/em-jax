import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from .estep import subject_laplace_fit
from .mstep import mstep
from .structs import EMFit


class _LiveStatus:
    """Prints a periodic status block that replaces the previous one in
    place, rather than scrolling, so you can watch a fit converge without
    the terminal filling up with old iterations.

    Args:
        enabled: if False, update() is a no-op.

    Notes:
        Detects the display environment and picks the right mechanism
        automatically, falling back to plain sequential printing when
        output isn't actually interactive (e.g. captured by a test runner
        or redirected to a file), so no escape codes leak into logs.
        Three modes:

        - a real Jupyter/IPython kernel (ZMQInteractiveShell): use
          IPython.display.clear_output(wait=True), the intended mechanism
          for a notebook's rich output area (ANSI codes aren't reliably
          honored there).
        - an interactive terminal (including a plain IPython terminal
          session, which isn't a notebook kernel but is still an ordinary
          ANSI-capable terminal): move the cursor up over the previous
          block and erase to end of screen, then print the new one in its
          place.
        - anything else: plain print, one block after another, no
          clearing.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.mode = self._detect() if enabled else "plain"
        self._nlines = 0

    @staticmethod
    def _detect():
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell":
                return "ipython"
        except Exception:
            pass
        try:
            if sys.stdout.isatty():
                return "tty"
        except Exception:
            pass
        return "plain"

    def update(self, text):
        if not self.enabled:
            return
        if self.mode == "ipython":
            from IPython.display import clear_output
            clear_output(wait=True)
            print(text)
        elif self.mode == "tty":
            if self._nlines:
                sys.stdout.write(f"\033[{self._nlines}A\033[J")
            print(text)
            sys.stdout.flush()
            self._nlines = text.count("\n") + 1
        else:
            print(text)


def _warn_single_device():
    warnings.warn(
        "pyem: only 1 JAX device visible, which largely prevents "
        "parallelization of subject fits across CPU cores. To use "
        "multiple CPU cores, set the environment variable "
        "XLA_FLAGS=--xla_force_host_platform_device_count=N (N <= your "
        "core count) before jax is imported -- e.g. "
        "os.environ[\"XLA_FLAGS\"] = \"--xla_force_host_platform_device_count=N\" "
        "as the first line of a script, or an early notebook cell run "
        "before any `import jax` (no kernel restart needed).",
        stacklevel=3,
    )


def _pad_and_reshape(arr, ndevices, per_device, npad):
    if npad:
        arr = jnp.concatenate([arr, arr[:npad]], axis=0)
    return arr.reshape((ndevices, per_device) + arr.shape[1:])


def _pad_and_reshape_data(data, ndevices, per_device, npad):
    return tuple(_pad_and_reshape(a, ndevices, per_device, npad) for a in data)


def _make_singledevice_fit(data, valid_mask, likfun, max_steps, skewcorrect, maxcorr):
    ndata = len(data)
    jitted = jax.jit(
        jax.vmap(subject_laplace_fit, in_axes=(0, 0, None, None, (0,) * ndata, 0, None, None, None, None)),
        static_argnums=(6, 7, 8),
    )

    def batched_fit(x, mu, inv_sigma, logdet_sigma):
        return jitted(x, mu, inv_sigma, logdet_sigma, data, valid_mask, likfun, max_steps,
                      skewcorrect, maxcorr)

    return batched_fit


def _make_multidevice_fit(ndevices, data, valid_mask, likfun, max_steps, skewcorrect, maxcorr):
    """Builds a batched per-subject Laplace-fit function parallelized
    across ndevices JAX devices.

    Args:
        ndevices: number of devices to use (clamped to nsub if smaller).
        data: tuple of (nsub, maxtrials) per-trial data arrays.
        valid_mask: (nsub, maxtrials) bool array.
        likfun: the model's negative-log-likelihood function.
        max_steps: maximum LBFGS iterations per subject.
        skewcorrect: whether to apply the skewness correction.
        maxcorr: maximum allowed correction size, used when
            skewcorrect=True.

    Returns:
        batched_fit(x, mu, inv_sigma, logdet_sigma) -> (x, l, h,
        hessian_fallback, skew_fallback), each stacked over all nsub
        subjects.

    Notes:
        Parallelizes at subject-chunk granularity via pmap: each device
        runs its own independent, unsynchronized vmap over its local chunk
        of subjects, rather than every device synchronizing on one batched
        op per trial across ALL subjects (which is what plain vmap alone
        would do). The former makes much better use of multiple cores,
        since each subject's likelihood is an internally-sequential scan
        over trials with no need to communicate across subjects at all.
    """
    ndata = len(data)
    nsub = data[0].shape[0]
    ndevices = min(ndevices, nsub)
    per_device = -(-nsub // ndevices)  # ceil division
    npad = per_device * ndevices - nsub
    devices = jax.devices()[:ndevices]

    data_dev = _pad_and_reshape_data(data, ndevices, per_device, npad)
    vm_dev = _pad_and_reshape(valid_mask, ndevices, per_device, npad)

    def per_device_fit(x_chunk, mu_chunk, inv_sigma, logdet_sigma, data_chunk, vm_chunk):
        return jax.vmap(subject_laplace_fit,
                         in_axes=(0, 0, None, None, (0,) * ndata, 0, None, None, None, None))(
            x_chunk, mu_chunk, inv_sigma, logdet_sigma, data_chunk, vm_chunk, likfun, max_steps,
            skewcorrect, maxcorr)

    pmapped = jax.pmap(per_device_fit, in_axes=(0, 0, None, None, (0,) * ndata, 0), devices=devices)

    def batched_fit(x, mu, inv_sigma, logdet_sigma):
        x_dev = _pad_and_reshape(x, ndevices, per_device, npad)
        mu_dev = _pad_and_reshape(mu, ndevices, per_device, npad)
        x_out, l_out, h_out, hfb_out, sfb_out = pmapped(x_dev, mu_dev, inv_sigma, logdet_sigma,
                                                          data_dev, vm_dev)

        # pmap's sharded outputs can't be reshaped/sliced and fed straight
        # back into the next pmap call -- the resulting sharding annotation
        # doesn't match what pmap expects on its next call. A clean host
        # round-trip in between avoids that; this is the only real overhead
        # this approach adds over plain vmap.
        def collect(out, shape_tail=()):
            return jnp.array(jax.device_get(out).reshape((-1,) + shape_tail)[:nsub])

        x_flat = collect(x_out, (x.shape[-1],))
        l_flat = collect(l_out)
        h_flat = collect(h_out, (h_out.shape[-2], h_out.shape[-1]))
        hfb_flat = collect(hfb_out)
        sfb_flat = collect(sfb_out)
        return x_flat, l_flat, h_flat, hfb_flat, sfb_flat

    return batched_fit


def _normalize_startbetas(startbetas, nreg, nparam):
    """Reshapes startbetas to (nreg, nparam) when given as a lower-rank
    shorthand.

    Args:
        startbetas: a (nreg, nparam) matrix, or (whenever nreg==1 or
            nparam==1) a plain vector or, if both are 1, a bare scalar.
        nreg: number of regressors.
        nparam: number of subject-level parameters.

    Returns:
        startbetas reshaped to (nreg, nparam).

    Notes:
        Left alone (and will error naturally downstream if the shape is
        actually wrong) whenever neither dimension is 1, since a bare
        vector's meaning would be ambiguous there.
    """
    arr = jnp.asarray(startbetas)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        if nreg == 1:
            return arr.reshape(1, nparam)
        if nparam == 1:
            return arr.reshape(nreg, 1)
    return arr


def _normalize_startsigma(startsigma, full):
    """Normalizes startsigma to match the shape em_fit's inner loop expects
    for full/diagonal covariance.

    Args:
        startsigma: a (nparam,) vector of starting variances, or an
            (nparam, nparam) matrix.
        full: if True, return a full (nparam, nparam) matrix; if False, a
            diagonal (nparam,) vector.

    Returns:
        startsigma normalized to match full.

    Notes:
        A plain (nparam,) vector is accepted regardless of full (and
        expanded to a diagonal matrix when full=True), so a vector is
        always the easy thing to hand in, even for a full/dense starting
        covariance, rather than needing to build the matrix yourself.
    """
    if startsigma.ndim == 1:
        return jnp.diag(startsigma) if full else startsigma
    return startsigma if full else jnp.diag(startsigma)


def em_fit(model, startbetas, startsigma, emtol=1e-4, maxiter=200, max_steps=256, startx=None,
           full=False, skewcorrect=False, skewcorrect_maxcorr=1.0, quiet=10):
    """Fits a hierarchical model via Expectation-Maximization: alternates a
    per-subject E-step (MAP + Laplace covariance under the current
    group-level prior) with a closed-form M-step (updating the group-level
    betas/sigma from the subjects' fits), until the group-level parameters
    stop changing by more than emtol between iterations.

    Args:
        model: an EMModel (X, data, valid_mask, likfun, nparam, ...).
        startbetas: a (nreg, nparam) matrix in general, but a plain
            (nparam,) vector (or, if nreg==nparam==1, a bare scalar) is
            accepted too whenever nreg==1 or nparam==1, since there's only
            one sensible way to interpret it in that case -- see
            _normalize_startbetas.
        startsigma: a plain (nparam,) vector of starting variances
            (accepted regardless of full -- expanded to a diagonal
            (nparam, nparam) matrix when full=True), or an already-
            (nparam, nparam) matrix (full still decides how it's used,
            taking its diagonal if full=False).
        emtol: convergence tolerance on the relative change in the packed
            group-level parameters between outer iterations.
        maxiter: maximum number of outer EM iterations.
        max_steps: maximum LBFGS iterations for each subject's MAP fit.
        startx: optionally warm-starts the per-subject solve from
            something other than X @ startbetas -- used by loocv to
            warm-start each leave-one-out refit from the full fit's own x.
        full: if False (default), fits a diagonal group covariance; if
            True, a full/dense one.
        skewcorrect: whether to apply the skewness correction (see
            subject_laplace_fit).
        skewcorrect_maxcorr: maximum allowed correction size, used when
            skewcorrect=True.
        quiet: 0 is fully silent; N>0 prints the current betas/sigma/
            change every N outer iterations (and on the final one).

    Returns:
        An EMFit. Its sigma comes back as a (nparam,) vector or
        (nparam, nparam) matrix matching full, not necessarily matching
        whatever shape startsigma was given in. It also carries total
        hessian_fallback/skew_fallback counts, summed across all outer
        iterations.

    Notes:
        Direct port of EM.jl's em(). The E-step is parallelized across
        whatever JAX devices are visible (see _make_multidevice_fit); if
        only one is visible, falls back to a single jitted vmap and warns
        (see _warn_single_device) -- same numerics either way, just faster
        with more devices.
    """
    X = model.X
    data = model.data
    valid_mask = model.valid_mask
    likfun = model.likfun
    nsub, nreg = X.shape
    startbetas = _normalize_startbetas(startbetas, nreg, model.nparam)
    startsigma = _normalize_startsigma(startsigma, full)

    ndevices = jax.local_device_count()
    if ndevices > 1:
        batched_fit = _make_multidevice_fit(ndevices, data, valid_mask, likfun, max_steps,
                                             skewcorrect, skewcorrect_maxcorr)
    else:
        _warn_single_device()
        batched_fit = _make_singledevice_fit(data, valid_mask, likfun, max_steps,
                                              skewcorrect, skewcorrect_maxcorr)

    betas = startbetas
    sigma = startsigma
    x = X @ betas if startx is None else startx

    def packed(betas, sigma):
        return jnp.concatenate([betas.flatten(), sigma.flatten()])

    prev = packed(betas, sigma)
    h = None
    l = None
    total_hessian_fallback = 0
    total_skew_fallback = 0
    status = _LiveStatus(enabled=bool(quiet))

    for it in range(1, maxiter + 1):
        mu = X @ betas
        sigma_full = sigma if full else jnp.diag(sigma)
        inv_sigma = jnp.linalg.inv(sigma_full)
        logdet_sigma = jnp.linalg.slogdet(sigma_full)[1]

        x, l, h, hfb, sfb = batched_fit(x, mu, inv_sigma, logdet_sigma)
        total_hessian_fallback += int(jnp.sum(hfb))
        total_skew_fallback += int(jnp.sum(sfb))

        betas, sigma = mstep(X, x, h, nreg, full=full)

        cur = packed(betas, sigma)
        change = float(jnp.max(jnp.abs((cur - prev) / prev)))
        prev = cur
        done = change < emtol

        if quiet and (done or it % quiet == 0):
            status.update(
                f"iter: {it}\n"
                f"betas: {np.round(np.array(betas), 4)}\n"
                f"sigma: {np.round(np.array(sigma), 4)}\n"
                f"change: {change:.6g}"
            )

        if done:
            break

    if quiet:
        if total_hessian_fallback > 0:
            print(f"Hessian fallback (non-PD Laplace Hessian, used group-level prior covariance "
                  f"instead): {total_hessian_fallback} / {it * nsub} subject-iterations")
        if skewcorrect and total_skew_fallback > 0:
            print(f"skew-correction fallback: {total_skew_fallback} / {it * nsub} subject-iterations")

    return EMFit(betas, sigma, x, l, h, model, iterations=it,
                 hessian_fallback=total_hessian_fallback, skew_fallback=total_skew_fallback)
