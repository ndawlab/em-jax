"""EMModel/EMFit/EMErrors: the model specification, fit result, and
standard-error result objects, each with a pretty-printed __repr__ so a
raw dict of arrays isn't the only way to inspect them -- this matters most
for EMErrors, where the ses/pvalues arrays alone have no labels attached
and aren't interpretable without a formatted table. Direct ports of
EM.jl's structs of the same names, including their Base.show formats."""
import numpy as np
import jax.numpy as jnp

from .data import _pad_by_subject


def _default_names(n, prefix):
    return [f"{prefix} {i + 1}" for i in range(n)]


def _fill_names(names, n, prefix):
    names = list(names) if names else []
    if len(names) < n:
        names = names + _default_names(n, prefix)[len(names):]
    return names


class EMModel:
    """Bundles a model specification: design matrix, per-trial data, the
    likelihood function, and dimension/naming metadata.

    Args:
        df: long-format DataFrame with one row per trial; must have a
            subject column (sub_col) and one column per entry in cols.
        X: (nsub, nreg) design matrix, or a plain (nsub,) vector for an
            intercept-only design (no covariates), reshaped to (nsub, 1)
            automatically. Its rows must be in subject order matching
            subs (see below).
        cols: names of df's per-trial data columns that likfun needs (e.g.
            (choices, rewards) for qlik/jianlik, (ch1, ch2, mn, st) for
            seqlik).
        sub_col: name of df's subject-id column.
        subs: explicit list of subject ids (any dtype, e.g. strings),
            fixing the row order shared by X and the padded per-subject
            data. Defaults to sorted(df[sub_col].unique()) if not given --
            pass this explicitly whenever X is built from a separate
            subject-level table, so both are guaranteed to agree on
            subject order.
        likfun: the model's negative-log-likelihood function.
        nparam: number of subject-level parameters.
        reg_names: optional list of regressor names, for pretty-printing.
        param_names: optional list of parameter names, for pretty-printing.

    Attributes:
        X, data, valid_mask, subs, likfun, nparam, reg_names, param_names:
            as above (X always 2-D; data is a tuple of (nsub, maxtrials)
            arrays in cols order; reg_names/param_names always fully
            populated, defaulting unnamed entries to "Reg i"/"Param i").
        nsub, nreg: X's shape.

    Notes:
        Direct port of Julia's EMModel, generalized to build the padded
        per-subject arrays from a long-format DataFrame directly. See
        from_arrays for the lower-level, array-based constructor (used
        internally, e.g. by loocv's leave-one-out refits).
    """

    def __init__(self, df, X, cols=("c", "r"), sub_col="sub", subs=None, *,
                 likfun, nparam, reg_names=None, param_names=None):
        if subs is None:
            subs = sorted(df[sub_col].unique())
        padded, valid_mask = _pad_by_subject(df, subs, sub_col=sub_col, cols=cols)
        self._init_arrays(X, *(padded[c] for c in cols), valid_mask=valid_mask, likfun=likfun,
                           nparam=nparam, reg_names=reg_names, param_names=param_names)
        self.subs = subs

    @classmethod
    def from_arrays(cls, X, *data, valid_mask, likfun, nparam, reg_names=None, param_names=None):
        """Builds an EMModel directly from already-padded arrays, bypassing
        the DataFrame/padding step.

        Args:
            X: (nsub, nreg) design matrix, or a plain (nsub,) vector.
            *data: however many (nsub, maxtrials) per-trial data arrays
                likfun needs.
            valid_mask: (nsub, maxtrials) bool array marking real
                (non-padding) trials.
            likfun: the model's negative-log-likelihood function.
            nparam: number of subject-level parameters.
            reg_names: optional list of regressor names.
            param_names: optional list of parameter names.

        Returns:
            An EMModel with subs=None (no subject ids are tracked in this
            path).

        Notes:
            Most users should construct EMModel from a DataFrame instead;
            this is for internal use (e.g. loocv's leave-one-out refits)
            and array-level testing against reference fixtures.
        """
        self = cls.__new__(cls)
        self._init_arrays(X, *data, valid_mask=valid_mask, likfun=likfun, nparam=nparam,
                           reg_names=reg_names, param_names=param_names)
        self.subs = None
        return self

    def _init_arrays(self, X, *data, valid_mask, likfun, nparam, reg_names, param_names):
        X = jnp.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.X = X
        self.data = data
        self.valid_mask = valid_mask
        self.likfun = likfun
        self.nparam = nparam
        self.nsub, self.nreg = X.shape
        self.reg_names = _fill_names(reg_names, self.nreg, "Reg")
        self.param_names = _fill_names(param_names, self.nparam, "Param")

    def __repr__(self):
        likfun_name = getattr(self.likfun, "__name__", repr(self.likfun))
        return (f"EMModel(nsub={self.nsub}, nreg={self.nreg}, nparam={self.nparam}, "
                f"likfun={likfun_name}, reg_names={self.reg_names}, "
                f"param_names={self.param_names})")


def _truncate(s, n):
    return s if len(s) <= n else s[:n - 3] + "..."


def _print_limited_matrix(mat, row_names, col_names, max_rows=6, max_cols=6):
    mat = np.array(mat)
    r, c = mat.shape
    show_c = min(c, max_cols)
    lines = ["".ljust(12) + "".join(f" {_truncate(col_names[j], 11):<11}" for j in range(show_c))
              + (f" {'...':<11}" if c > show_c else "")]
    show_r = min(r, max_rows)
    for i in range(show_r):
        row = f"{_truncate(row_names[i], 11) + ':':<12}"
        row += "".join(f" {mat[i, j]:<11.4f}" for j in range(show_c))
        if c > show_c:
            row += f" {'...':<11}"
        lines.append(row)
    if r > show_r:
        row = "...".ljust(12) + "".join(f" {'...':<11}" for _ in range(show_c))
        if c > show_c:
            row += f" {'...':<11}"
        lines.append(row)
    return "\n".join(lines)


class EMFit:
    """Result of em_fit: group-level betas/sigma, per-subject fits, and a
    reference to the EMModel it came from.

    Args:
        betas: (nreg, nparam) group-level regression coefficients.
        sigma: (nparam,) diagonal or (nparam, nparam) full group
            covariance.
        x: (nsub, nparam) per-subject parameter estimates.
        l: (nsub,) per-subject negative log-likelihoods.
        h: (nsub, nparam, nparam) per-subject Laplace covariances.
        model: the EMModel this fit was produced from.
        iterations: number of outer EM iterations taken, or None.
        hessian_fallback: count of subjects whose Hessian fell back to the
            prior precision (see safe_laplace_hessian).
        skew_fallback: count of subjects whose skewness correction was
            discarded (see subject_laplace_fit).

    Attributes:
        Same names as the constructor args above.

    Notes:
        Direct port of Julia's EMFit, including its formatted __repr__.
        Supports 5-element destructuring for plain array-based code:
        betas, sigma, x, l, h = fit.
    """

    def __init__(self, betas, sigma, x, l, h, model, iterations=None,
                 hessian_fallback=0, skew_fallback=0):
        self.betas = betas
        self.sigma = sigma
        self.x = x
        self.l = l
        self.h = h
        self.model = model
        self.iterations = iterations
        self.hessian_fallback = hessian_fallback
        self.skew_fallback = skew_fallback

    def __iter__(self):
        return iter((self.betas, self.sigma, self.x, self.l, self.h))

    def __repr__(self):
        nreg, nparam = self.betas.shape
        nsub = self.model.nsub
        reg_names, param_names = self.model.reg_names, self.model.param_names

        sigma = np.array(self.sigma)
        sigma_full = np.diag(sigma) if sigma.ndim == 1 else sigma

        lines = [
            "EM model estimation results:",
            "-" * 56,
            f"Subjects: {nsub}",
            f"Regressors: {nreg}",
            f"Parameters: {nparam}",
            "-" * 56,
            "Group-level Coefficients (betas):",
            _print_limited_matrix(self.betas, reg_names, param_names),
            "",
            "Group-level Covariance (sigma):",
            _print_limited_matrix(sigma_full, param_names, param_names),
            "",
            "Subject-level results:",
            f"  x: {nsub} x {nparam} subject-level parameters",
            f"  l: {nsub} subject-level negative log-likelihoods (sum: {float(np.sum(np.array(self.l))):.4f})",
            f"  h: {nsub} x {nparam} x {nparam} subject-level covariances",
            "-" * 56,
        ]
        return "\n".join(lines)


class EMErrors:
    """Result of emerrors: standard errors, p-values, and the group-level
    covariance matrix over the betas, plus a reference to the parent EMFit.

    Args:
        ses: (nreg*nparam,) standard errors for the flattened betas.
        pvalues: (nreg*nparam,) two-sided p-values.
        covmtx: (nreg*nparam, nreg*nparam) covariance matrix over the
            flattened betas.
        fit: the parent EMFit.

    Attributes:
        Same names as the constructor args above.

    Notes:
        Direct port of Julia's EMErrors, including its formatted
        regression-table __repr__ (with significance stars) -- use the
        printed table rather than the raw ses/pvalues arrays, which have
        no labels attached. Supports 3-element destructuring:
        ses, pvalues, covmtx = errs.
    """

    def __init__(self, ses, pvalues, covmtx, fit):
        self.ses = ses
        self.pvalues = pvalues
        self.covmtx = covmtx
        self.fit = fit

    def __iter__(self):
        return iter((self.ses, self.pvalues, self.covmtx))

    def __repr__(self):
        betas = np.array(self.fit.betas)
        nreg, nparam = betas.shape
        reg_names, param_names = self.fit.model.reg_names, self.fit.model.param_names
        ses = np.array(self.ses)
        pvalues = np.array(self.pvalues)

        header = (f"{'Regressor':<16} {'Parameter':<16} {'Estimate':>10} {'Std.Error':>12} "
                  f"{'t-value':>10}    {'p-value':<7}")
        lines = ["=" * 72, header, "=" * 72]
        for r in range(nreg):
            for p in range(nparam):
                idx = r * nparam + p
                est = betas[r, p]
                se = ses[idx]
                pval = pvalues[idx]
                tval = est / se if se > 0 else float("nan")
                p_str = "<0.0001" if pval < 0.0001 else f"{pval:.4f}"
                if pval < 0.001:
                    stars = "***"
                elif pval < 0.01:
                    stars = "**"
                elif pval < 0.05:
                    stars = "*"
                elif pval < 0.1:
                    stars = "."
                else:
                    stars = " "
                lines.append(f"{reg_names[r]:<16} {param_names[p]:<16} {est:>10.4f} {se:>12.4f} "
                              f"{tval:>10.2f}    {p_str:<7} {stars}")
        lines.append("=" * 72)
        lines.append("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        return "\n".join(lines)
