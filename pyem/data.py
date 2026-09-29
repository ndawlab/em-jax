import numpy as np
import jax.numpy as jnp


def _pad_by_subject(df, subs, sub_col="sub", cols=("c", "r")):
    """Converts a long-format DataFrame with ragged per-subject trial counts
    into fixed-shape, per-subject-padded arrays, in the given subject
    order.

    Args:
        df: long-format DataFrame with one row per trial; must have a
            subject column (sub_col) and one column per entry in cols.
        subs: list of subject ids fixing the row order of the returned
            arrays.
        sub_col: name of the subject-id column.
        cols: names of the per-trial data columns to pad (e.g. choices and
            rewards).

    Returns:
        (padded, valid_mask): padded is a dict mapping each name in cols
        to an (nsub, maxtrials) int64 array; valid_mask is an
        (nsub, maxtrials) bool array marking real (non-padding) trials.
        Both are in the row order given by subs.

    Notes:
        Used internally by EMModel's DataFrame constructor. A single
        groupby is used rather than filtering the whole frame once per
        subject, since the latter costs O(nsub^2) overall.
    """
    groups = dict(list(df.groupby(sub_col)))
    nsub = len(subs)
    maxtrials = max(len(groups[s]) for s in subs)

    padded = {c: np.zeros((nsub, maxtrials), dtype=np.int64) for c in cols}
    valid_mask = np.zeros((nsub, maxtrials), dtype=bool)

    for i, sub in enumerate(subs):
        sub_df = groups[sub]
        n = len(sub_df)
        for c in cols:
            padded[c][i, :n] = sub_df[c].to_numpy()
        valid_mask[i, :n] = True

    return (
        {c: jnp.array(padded[c]) for c in cols},
        jnp.array(valid_mask),
    )
