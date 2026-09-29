# pyem

A Python/JAX port of [EM.jl](https://github.com/ndawlab/em)'s
hierarchical-EM + Laplace-approximation model fitting, for people who'd
rather work in Python.

This means you can load your datasets in using pandas etc., and write
your likelihood function in Python, and fit a hierarchical model (with
covariates etc.) super fast without ever touching julia.

The bad news is that making this work (and it does work well) requires 
you use jax (rather than standard numpy). This is performant but much 
more diabolical than Julia. It's not too late to learn Julia! But if you
insist, you can use this instead. likelihoods.py has examples of what it
takes to write your likelihood in jax, which is a little idiosyncratic
but gets the job done.

## Install

```
pip install git+https://github.com/ndawlab/em-jax.git
```

To instead work on `pyem` itself (e.g. to run the test suite), clone the
repo and install it editable, with the `test` extra (just `pytest`):

```
git clone https://github.com/ndawlab/em-jax.git
cd em-jax
pip install -e ".[test]"
```

## Examples

see examples/example.py or example.ipynb

or, minimally:

```python
import jax
jax.config.update("jax_enable_x64", True)  # required -- JAX defaults to float32
import jax.numpy as jnp
import pyem as em

# data: long DataFrame with sub/c/r columns (one row per trial)
subs = sorted(data["sub"].unique())  # fixes the subject order X's rows must match
X = jnp.ones(len(subs))  # design matrix: intercept-only here, so just a vector -- see below

model = em.EMModel(data, X, subs=subs, likfun=em.qlik_nll, nparam=2,
                    reg_names=["Intercept"], param_names=["Temp", "LR"])  # names optional, for __repr__

startbetas = jnp.array([1.0, 0.0])     # one entry per parameter (again: only one regressor)
startsigma = jnp.array([5.0, 1.0])     # a plain (nparam,) vector works for either covariance shape

fit = em.em_fit(model, startbetas, startsigma)               # diagonal group covariance (the default)
# fit = em.em_fit(model, startbetas, startsigma, full=True)   # full/dense instead -- same startsigma
# skewcorrect=True, skewcorrect_maxcorr=1.0 also accepted
print(fit)                # EMFit's pretty-printer: labeled betas/sigma, subject-level summary
print(em.emerrors(fit))   # EMErrors's pretty-printer: an R-style regression table with
                           # significance stars -- use this, not the raw ses/pvalues arrays,
                           # which have no labels attached and aren't interpretable on their own
```

`subs` fixes the row order shared by `X` and the padded per-trial data
`EMModel` builds internally from `data`; it defaults to
`sorted(data["sub"].unique())` if omitted, but pass it explicitly whenever
`X` comes from a separate subject-level table (e.g. a covariate lookup
keyed by subject id), so both are guaranteed to agree on subject order --
subject ids don't need to be integers, e.g. arbitrary string ids work
the same way.

**Vector shorthand.** `X` is a `(nsub, nreg)` matrix and `startbetas` a
`(nreg, nparam)` matrix in general, but with only one regressor (an
intercept-only design, no covariates) there's only one sensible way to
read a plain vector, so `EMModel`/`em_fit` accept one: `X` as a bare
`(nsub,)` vector (reshaped to `(nsub, 1)` automatically), and `startbetas`
as a bare `(nparam,)` vector whenever `nreg==1` (or `(nreg,)` whenever
`nparam==1`, or a bare scalar if both are 1). `startsigma` already works
this way regardless of `full` (see above); see the commented-out
intercept-only variant in `examples/example.py`.

Every public name (`em_fit`, `emerrors`, `EMModel`, `qlik_nll`, `simq`,
...) lives at the top level of the `pyem` package -- `import pyem as em`
and call `em.whatever(...)`, rather than importing each piece from its own
submodule (`pyem.emloop`, `pyem.errors`, ...); those submodules are an
internal implementation detail, not part of the public interface.

The per-trial data columns (`c`/`r` above, read from `data` and passed
through to `likfun`) are however many arrays the likelihood needs.
`seqlik_nll` takes four instead of two -- pass `cols=("ch1","ch2","mn","st")`
to `EMModel`, and `likfun=seqlik_nll, nparam=6`.

`EMModel.from_arrays(X, *data, valid_mask=..., likfun=..., nparam=...)` is
a lower-level constructor that takes already-padded arrays directly,
bypassing the DataFrame/padding step -- used internally (e.g. by `loocv`'s
leave-one-out refits) and useful for validating against array-level
reference data; most users should construct `EMModel` from a DataFrame
instead.

**Progress.** `em_fit(..., quiet=10)` (the default) prints the current
betas/sigma/change every 10 outer iterations, replacing the previous
printout in place rather than scrolling, so you can watch convergence live
without the terminal filling up -- this works the same way in a plain
terminal or a Jupyter notebook, with no setup needed either way. Pass
`quiet=0` for silence, or `quiet=N` for a different interval.
`loocv(..., progress=True)` (also the default) shows a progress bar over
its `nsub` leave-one-out refits, since `nsub` is a real, known total there;
pass `progress=False` to silence it.

`emerrors`/`lml`/`ilaplace`/`ibic`/`iaic`/`loocv` all just take the `fit`
returned by `em_fit` rather than each needing several separate array
arguments -- `fit.model` carries everything (`X`, the data, `likfun`,
`reg_names`/`param_names`, ...) needed to compute or display them.
`EMFit`/`EMErrors` also support tuple-style destructuring if you'd rather
work with plain arrays: `betas, sigma, x, l, h = fit` and
`ses, pvalues, covmtx = emerrors(fit)`.

## Parallelism

JAX's CPU backend multithreads automatically within a single device
(nothing to set), but `em_fit`'s per-subject E-step gets real,
coarse-grained multi-core parallelism (one chunk of subjects per core, each
running independently, no synchronization until the whole batch is done)
only when more than one JAX device is visible. Set the environment
variable `XLA_FLAGS=--xla_force_host_platform_device_count=N` (N <= your
core count) **before `jax` is imported** -- it doesn't need to be set
before Python itself starts, just before the first `import jax`, e.g. as
`os.environ["XLA_FLAGS"] = "..."` at the top of a script. That also means
it's notebook-friendly: setting it in an early cell (before any cell has
imported `jax`) is enough, no kernel restart required. `em_fit` warns at
runtime if it doesn't see more than one device.

Without this set, per-subject fitting still runs, just with much more
modest CPU utilization: each subject's likelihood is a sequential
`lax.scan` over trials, so JAX can only parallelize the batch-of-subjects
dimension *around* that per-trial recurrence, and the per-trial
computation is usually too small to offset the dispatch/synchronization
overhead of doing that well. Expect a somewhat noticeable delay on the
first call to any given problem shape either way, from JIT compilation.

## Tests

```
pytest tests/
```

Validates against reference output from `EM.jl`, checked in under
`tests/fixtures/`.

## Writing a new likelihood

A likelihood is a plain function `(params, *data, valid_mask) ->
negative_log_likelihood`. The easiest way to write one is the
`@scan_likelihood` decorator, which hides all the `lax.scan` and
ragged-data-masking mechanics -- you write a `step_maker(params)` that
returns `(init_state, step)`, and `step(state, *trial_arrays) -> (new_state,
ll)` as if every trial were real and valid:

```python
import pyem as em

@em.scan_likelihood
def qlik_nll(params):
    beta = params[0]
    lr = 0.5 + 0.5 * erf(params[1] / jnp.sqrt(2.0))

    def step(Q, c, r):
        idx = c - 1
        ll = beta * Q[idx] - logsumexp(beta * Q)
        Q_new = Q.at[idx].set((1 - lr) * Q[idx] + r)
        return Q_new, ll

    return jnp.zeros(2), step
```

This is what `qlik_nll`/`jianlik_nll` actually are in `pyem/likelihoods.py`.
The decorator handles padding and missed-response trials the same way: a
trial is skipped (state unchanged, `ll=0`) if it's padding *or* if its first
data array is `<= 0`, exactly as if it had been deleted from the data.
Anything the model needs from earlier trials, such as the previous choice
for a perseveration term, just goes in `state`.

`seqlik_nll` is an example of both: it carries the previous first-stage
choice in its state, and since a real trial can still have its
*second*-stage response missing (`c1` present, `c2 <= 0`), it handles that
case inside `step` with an ordinary `jnp.where` (count `ll1` but not `ll2`,
leave the Q values unchanged).
