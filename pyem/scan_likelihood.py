import functools

import jax
import jax.numpy as jnp


def scan_likelihood(step_maker=None, *, track_prev=False):
    """Decorator that turns a per-trial step function into a full
    negative-log-likelihood function, hiding the lax.scan boilerplate and
    the ragged-data masking convention from the model author.

    Args:
        step_maker: a function params -> (init_state, step), where
            step(state, *trial_arrays) -> (new_state, ll) is called once
            per trial (state can be a single array or a pytree of several,
            e.g. (Q, lr) for a model with a second piece of scanned
            state). Write step as if every trial were real and valid; any
            special-casing a real trial with partial/missing later
            responses needs is ordinary model logic and belongs inside
            step itself (see seqlik_nll for an example), not something
            this decorator handles.
        track_prev: if True, step additionally receives the *raw* previous
            trial's first data array value as a second argument, before
            the trial's own data: step(state, prev, *trial_arrays) ->
            (new_state, ll). For models with a perseveration/sticky-choice
            term that needs "what was chosen last trial" as an input.

    Returns:
        A decorator producing nll(params, *data, valid_mask) -> scalar,
        matching every other likelihood in this package and ready to hand
        to em_fit/loocv as-is.

    Notes:
        A trial is skipped (state frozen to its pre-trial value, ll=0) if
        it's padding or the first data array is <= 0 -- state gets this
        uniform freeze-on-invalid treatment precisely because it's meant
        to represent "what's genuinely been learned so far," which a
        missing/padding trial must leave untouched.

        track_prev's prev value is deliberately NOT part of state and
        doesn't get the freeze-on-invalid treatment: it always advances to
        this trial's raw first array value, even 0 (a miss or padding), so
        perseveration correctly sees "no genuine previous choice" right
        after a gap, rather than reaching back through it to a stale one
        (seqlik_nll uses this for exactly that reason).

    Usage:

        @scan_likelihood
        def qlik_nll(params):
            beta = params[0]
            lr = 0.5 + 0.5 * erf(params[1] / jnp.sqrt(2.0))

            def step(Q, c, r):
                idx = c - 1
                ll = beta * Q[idx] - logsumexp(beta * Q)
                Q_new = Q.at[idx].set((1 - lr) * Q[idx] + r)
                return Q_new, ll

            return jnp.zeros(2), step
    """

    def decorator(step_maker):
        @functools.wraps(step_maker)
        def nll(params, *data, valid_mask):
            init_state, step = step_maker(params)
            prev0 = jnp.array(0, dtype=data[0].dtype)
            init_carry = (init_state, prev0) if track_prev else init_state

            def scan_step(carry, xs):
                *trial_arrays, valid = xs
                valid = valid & (trial_arrays[0] > 0)
                if track_prev:
                    state, prev = carry
                    new_state, ll = step(state, prev, *trial_arrays)
                else:
                    state = carry
                    new_state, ll = step(state, *trial_arrays)
                ll = jnp.where(valid, ll, 0.0)
                masked_state = jax.tree_util.tree_map(
                    lambda n, o: jnp.where(valid, n, o), new_state, state)
                new_carry = (masked_state, trial_arrays[0]) if track_prev else masked_state
                return new_carry, ll

            _, lls = jax.lax.scan(scan_step, init_carry, (*data, valid_mask))
            return -jnp.sum(lls)

        return nll

    if step_maker is not None:
        return decorator(step_maker)
    return decorator
