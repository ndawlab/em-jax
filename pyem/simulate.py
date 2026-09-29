"""Plain-numpy ports of Julia's simq/simseq/simjian, for generating synthetic
data matching qlik/seqlik/jianlik's generative models.
"""
import numpy as np
from scipy.special import erf


def simq(params, ntrials, rng=None):
    """Simulates choices and rewards for a Rescorla-Wagner Q-learning
    agent, matching qlik_nll's generative model.

    Args:
        params: [beta, lr_raw], as in qlik_nll.
        ntrials: number of trials to simulate.
        rng: numpy Generator, or None to create a fresh default one.

    Returns:
        (c, r): each length ntrials; c is 1-indexed choices in {1, 2}, r
        is rewards in {-1, 1}.
    """
    rng = np.random.default_rng() if rng is None else rng
    beta = params[0]
    lr = 0.5 + 0.5 * erf(params[1] / np.sqrt(2.0))

    c = np.zeros(ntrials, dtype=np.int64)
    r = np.zeros(ntrials, dtype=np.int64)
    Q = np.zeros(2)

    for i in range(ntrials):
        ps = np.exp(beta * Q)
        ps = ps / ps.sum()
        c[i] = int(rng.random() > ps[0]) + 1
        r[i] = 2 * round(rng.random()) - 1
        Q[c[i] - 1] = (1 - lr) * Q[c[i] - 1] + r[i]

    return c, r


def simjian(params, ntrials, rng=None):
    """Simulates choices and rewards for the hybrid learning agent,
    matching jianlik_nll's generative model.

    Args:
        params: [beta_raw, nu_raw], as in jianlik_nll.
        ntrials: number of trials to simulate.
        rng: numpy Generator, or None to create a fresh default one.

    Returns:
        (c, r): each length ntrials; c is 1-indexed choices in {1, 2}, r
        is rewards in {-1, 1}.
    """
    rng = np.random.default_rng() if rng is None else rng
    beta = 2 * params[0]
    nu = 0.5 + 0.5 * erf(params[1] / np.sqrt(2.0))

    c = np.zeros(ntrials, dtype=np.int64)
    r = np.zeros(ntrials, dtype=np.int64)
    Q = np.zeros(2)
    lr = 1.0

    for i in range(ntrials):
        ps = np.exp(beta * Q)
        ps = ps / ps.sum()
        c[i] = int(rng.random() > ps[0]) + 1
        r[i] = 2 * round(rng.random()) - 1
        delta = r[i] - Q[c[i] - 1]
        Q[c[i] - 1] += lr / 2 * delta
        lr = (1 - nu) * lr + nu * abs(delta)

    return c, r


def _softmaximum(a, b):
    p = 1.0 / (1.0 + np.exp(-5 * (a - b)))
    return p * a + (1 - p) * b


def simseq(params, ntrials, rng=None):
    """Simulates a two-step decision task agent, matching seqlik_nll's
    generative model.

    Args:
        params: [beta1m, beta1t0, beta1t1, beta2, lr_raw, ps], as in
            seqlik_nll.
        ntrials: number of trials to simulate.
        rng: numpy Generator, or None to create a fresh default one.

    Returns:
        (c1, s, c2, r): each length ntrials; c1 is first-stage choice in
        {1, 2}, s is second-stage state in {2, 3}, c2 is second-stage
        choice in {1, 2}, r is reward in {-1, 1}.
    """
    rng = np.random.default_rng() if rng is None else rng
    beta1m, beta1t0, beta1t1, beta2 = params[0], params[1], params[2], params[3]
    lr = 0.5 + 0.5 * erf(params[4] / np.sqrt(2.0))
    ps = params[5]

    c1 = np.zeros(ntrials, dtype=np.int64)
    c2 = np.zeros(ntrials, dtype=np.int64)
    r = np.zeros(ntrials, dtype=np.int64)
    s = np.zeros(ntrials, dtype=np.int64)

    Q0 = np.zeros((3, 2))
    Q1 = np.zeros(2)
    prevc = 0

    for i in range(ntrials):
        Qm = np.array([_softmaximum(Q0[1, 0], Q0[1, 1]), _softmaximum(Q0[2, 0], Q0[2, 1])])
        Qd = beta1m * Qm + beta1t0 * Q0[0, :] + beta1t1 * Q1
        if prevc > 0:
            Qd[prevc - 1] += ps

        cp = np.exp(Qd)
        cp = cp / cp.sum()
        c1[i] = int(rng.random() > cp[0]) + 1
        s[i] = int(rng.random() > (0.7 if c1[i] == 1 else 0.3)) + 2

        cp = np.exp(beta2 * Q0[s[i] - 1, :])
        cp = cp / cp.sum()
        c2[i] = int(rng.random() > cp[0]) + 1

        r[i] = 2 * round(rng.random()) - 1

        Q0[0, c1[i] - 1] = (1 - lr) * Q0[0, c1[i] - 1] + Q0[s[i] - 1, c2[i] - 1]
        Q1[c1[i] - 1] = (1 - lr) * Q1[c1[i] - 1] + r[i]
        Q0[s[i] - 1, c2[i] - 1] = (1 - lr) * Q0[s[i] - 1, c2[i] - 1] + r[i]

        prevc = c1[i]

    return c1, s, c2, r
