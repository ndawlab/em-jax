"""Targeted validation of seqlik_nll's genuine-missing-response handling
(as opposed to trailing padding, which is all the simseq-generated
reference_data_seqlik.csv fixture exercises): a mid-sequence c1 miss, and
a mid-sequence c1-present-but-c2-missing trial, plus a perseveration term
that must correctly reset after a miss. See
scripts/gen_reference_seqlik_missing.jl for the hand-crafted trials."""
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pandas as pd
import pytest

from pyem.data import _pad_by_subject
from pyem.likelihoods import seqlik_nll

FIXTURES = Path(__file__).parent / "fixtures"


def test_seqlik_missing_responses_match_julia():
    df = pd.read_csv(FIXTURES / "reference_data_seqlik_missing.csv")
    subs = sorted(df["sub"].unique())
    padded, valid_mask = _pad_by_subject(df, subs, cols=("ch1", "ch2", "mn", "st"))

    with open(FIXTURES / "reference_nll_seqlik_missing.txt") as f:
        rows = [line.strip().split(",") for line in f]

    for i, row in enumerate(rows):
        params = jnp.array([float(v) for v in row[1:7]])
        nll_ref = float(row[7])
        nll = seqlik_nll(params, padded["ch1"][i], padded["ch2"][i], padded["mn"][i], padded["st"][i],
                          valid_mask=valid_mask[i])
        assert float(nll) == pytest.approx(nll_ref, abs=1e-7)
