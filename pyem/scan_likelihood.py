import functools

import jax
import jax.numpy as jnp


def scan_likelihood(step_maker=None):
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

    Returns:
        A decorator producing nll(params, *data, valid_mask) -> scalar,
        matching every other likelihood in this package and ready to hand
        to em_fit/loocv as-is.

    Notes:
        A trial is skipped (state unchanged, ll=0) if it's padding or the
        first data array is <= 0, exactly as if it had been deleted from
        the data. Anything the model needs from earlier trials (e.g. the
        previous choice, for perseveration) should be carried in state.

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

            def scan_step(state, xs):
                *trial_arrays, valid = xs
                valid = valid & (trial_arrays[0] > 0)
                new_state, ll = step(state, *trial_arrays)
                ll = jnp.where(valid, ll, 0.0)
                new_state = jax.tree_util.tree_map(
                    lambda n, o: jnp.where(valid, n, o), new_state, state)
                return new_state, ll

            _, lls = jax.lax.scan(scan_step, init_state, (*data, valid_mask))
            return -jnp.sum(lls)

        return nll

    if step_maker is not None:
        return decorator(step_maker)
    return decorator
