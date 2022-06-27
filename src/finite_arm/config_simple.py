"""Specify the jobs to run via config file.

A simple experiment comparing Thompson sampling to greedy algorithm. Finite
armed bandit with 3 arms. Greedy algorithm premature and suboptimal
exploitation.
See Figure 3 from https://arxiv.org/abs/1707.02038
"""

import collections
import functools

import pytest

from base.config_lib import Config
from base.experiment import BaseExperiment
from finite_arm.agent_finite import FiniteBernoulliBanditEpsilonGreedy
from finite_arm.agent_finite import FiniteBernoulliBanditTS
from finite_arm.env_finite import FiniteArmedBernoulliBandit


def get_config(probs: list[float], n_steps: int, n_seeds: int):
    """Generates the config for the experiment."""

    # check all probs between 0 and 1:p
    assert all(
        0 <= prob and prob <= 1 for prob in probs
    ), 'Not all probabilities are between 0 and 1'

    # check all pro
    n_arm = len(probs)

    name = "finite_simple"
    # n_arm = 3
    n_arm = 2
    agents = collections.OrderedDict(
        [
            (
                "greedy",
                functools.partial(FiniteBernoulliBanditEpsilonGreedy, n_arm),
            ),
            ("ts", functools.partial(FiniteBernoulliBanditTS, n_arm)),
        ]
    )
    # probs = [0.7, 0.8, 0.9]
    probs = [0.2, 0.3]
    # probs = [0.02, 0.03]
    # probs = [0.0022, 0.0024, 0.0026, 0.0028, 0.0030]
    environments = collections.OrderedDict(
        [("env", functools.partial(FiniteArmedBernoulliBandit, probs))]
    )
    experiments = collections.OrderedDict([(name, BaseExperiment)])
    n_steps = 10000
    n_seeds = 10000
    config = Config(name, agents, environments, experiments, n_steps, n_seeds)
    return config


def test_get_config_incorrect_probs():
    with pytest.raises(Exception):
        get_config([-1, 0, 0.5], 1, 1)
        get_config([-0, 0.5, 1.4], 1, 1)
