"""Specify the jobs to run via config file.

A simple experiment comparing Thompson sampling to greedy algorithm. Finite
armed bandit with 3 arms. Greedy algorithm premature and suboptimal
exploitation.
See Figure 3 from https://arxiv.org/abs/1707.02038
"""

import pytest

from .config_simple import get_config


def test_get_config_incorrect_probs():
    with pytest.raises(Exception):
        get_config([-1, 0, 0.5])
        get_config([-0, 0.5, 1.4])
