import jax.numpy as jnp
from jax.scipy.special import erf
from jax.scipy.special import logsumexp

from .scan_likelihood import scan_likelihood


@scan_likelihood
def qlik_nll(params):
    """Negative log-likelihood of a subject's choices and rewards under a
    basic Rescorla-Wagner Q-learning model. (The @scan_likelihood decorator
    turns this into the callable described below.)

    Args:
        params: [beta, lr_raw]. beta is the softmax inverse temperature;
            lr_raw is the learning rate before being squashed to (0, 1) via
            the unit-normal CDF, so that a unit-normal prior on lr_raw
            corresponds to a uniform prior on the learning rate itself.
        (step function is also called with per-trial c, r)

    Returns:
        Scalar total negative log-likelihood.

    Notes:
        The likelihood function, first, translates the parameters, and,
        second, defines a step function that takes the learner state (here Q) 
        and data (here: c, r) for each trial, returns the state and per-trial
        choice likelihood. Like the inside of a for loop over trials.

        It is decorated with @scan_likelihood which turns it into the for loop.

        The scan wrapper omits missing or padded trials: this body is written as
        if every trial were valid.

        The end of the function returns the initial state (ie Q = jnp.zeros(2)) 
        and the step function.
    """
    beta = params[0]
    lr = 0.5 + 0.5 * erf(params[1] / jnp.sqrt(2.0))

    def step(Q, c, r):
        idx = c - 1  # c is 1 or 2
        ll = beta * Q[idx] - logsumexp(beta * Q)
        Q_new = Q.at[idx].set((1 - lr) * Q[idx] + r)
        return Q_new, ll

    return jnp.zeros(2), step


@scan_likelihood
def jianlik_nll(params):
    """Negative log-likelihood for a simplified version of the hybrid 
    learning model of Li et al. (Nat Neurosci, 2011): a delta-rule 
    update whose learning rate itself adapts to the size of recent 
    prediction errors.

    Args:
        params: [beta_raw, nu_raw]. beta_raw is half the softmax inverse
            temperature (beta = 2 * beta_raw); nu_raw is the meta-learning
            rate before being squashed to (0, 1) via the unit-normal CDF.
        (step function is also called with per-trial c, r)

    Returns:
        Scalar total negative log-likelihood.

    Notes:
        See qlik for likfun conventions
    """
    beta = 2 * params[0] # scaled to match qlik
    nu = 0.5 + 0.5 * erf(params[1] / jnp.sqrt(2.0))

    def step(state, c, r):
        Q, lr = state
        idx = c - 1
        ll = beta * Q[idx] - logsumexp(beta * Q)
        delta = r - Q[idx]
        Q_new = Q.at[idx].set(Q[idx] + lr / 2 * delta)
        lr_new = (1 - nu) * lr + nu * jnp.abs(delta)
        return (Q_new, lr_new), ll

    return (jnp.zeros(2), jnp.array(1.0)), step


def _softmaximum(a, b):
    p = 1.0 / (1.0 + jnp.exp(-5 * (a - b)))
    return p * a + (1 - p) * b


@scan_likelihood(track_prev=True)
def seqlik_nll(params):
    """Negative log-likelihood for the two-step decision task model of
    Gillan et al. (eLife, 2015).

    Args:
        params: [beta1m, beta1t0, beta1t1, beta2, lr_raw, ps]: model-based
            and two model-free weights on the first-stage decision, the
            second-stage inverse temperature, the learning rate before
            being squashed to (0, 1) via the unit-normal CDF, and the
            perseveration weight.
        the likfun is also called with 
            c1 (first stage choice: 1 or 2)
            c2 (second stage choice: 1 or 2),
            r (reward),
            s (second stage state: 2 or 3)
        

    Returns:
        Scalar total negative log-likelihood.

    Notes:
        Compared to the earlier examples, this takes more info per trial
        (c1, c2, r, s) and also the previous choice (track_prev above)
        The ommission of skipped trials is more complicated: this function
        is called if there is a c1 but must check if c2 is missing.
    """
    beta1m, beta1t0, beta1t1, beta2 = params[0], params[1], params[2], params[3]
    lr = 0.5 + 0.5 * erf(params[4] / jnp.sqrt(2.0))
    ps = params[5]

    def step(state, prevc, c1, c2, r, s):
        Q0, Q1 = state
        idx1 = c1 - 1
        sidx = s - 1

        Qm = jnp.array([_softmaximum(Q0[1, 0], Q0[1, 1]), _softmaximum(Q0[2, 0], Q0[2, 1])])
        Qd = beta1m * Qm + beta1t0 * Q0[0, :] + beta1t1 * Q1
        safe_prevc = jnp.where(prevc > 0, prevc - 1, 0)
        Qd = jnp.where(prevc > 0, Qd.at[safe_prevc].add(ps), Qd)
        ll1 = Qd[idx1] - logsumexp(Qd)

        # second-stage response can be missed even on a real trial -- this
        # is the model's own logic, not generic masking (see docstring)
        valid2 = c2 > 0
        idx2 = jnp.where(valid2, c2 - 1, 0)
        ll2 = jnp.where(valid2, beta2 * Q0[sidx, idx2] - logsumexp(beta2 * Q0[sidx, :]), 0.0)

        old_Q0_s_c2 = Q0[sidx, idx2]
        Q0_upd = Q0.at[0, idx1].set((1 - lr) * Q0[0, idx1] + old_Q0_s_c2)
        Q0_upd = Q0_upd.at[sidx, idx2].set((1 - lr) * old_Q0_s_c2 + r)
        Q1_upd = Q1.at[idx1].set((1 - lr) * Q1[idx1] + r)

        Q0_new = jnp.where(valid2, Q0_upd, Q0)
        Q1_new = jnp.where(valid2, Q1_upd, Q1)

        return (Q0_new, Q1_new), ll1 + ll2

    return (jnp.zeros((3, 2)), jnp.zeros(2)), step
