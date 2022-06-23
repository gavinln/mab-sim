"""Run an experiment locally from a config file.

We suggest that you use batch_runner.py for large scale experiments.
However, if you just want to play around with a smaller scale experiment then
this script can be useful.

The config file defines the selection of agents/environments/seeds that we want
to run. This script then runs through the first `N_JOBS` job_id's and then
collates the results for a simple plot.

Effectively, this script combines:
  - running `batch_runner.py` for several job_id
  - running `batch_analysis.py` to collate the data written to .csv

This is much simpler and fine for small sweeps, but is not scalable to large
parallel evaluations.

"""
import importlib
import os
import sys

from base import config_lib

import numpy as np
import pandas as pd
import plotnine as gg

from finite_arm import config_simple


def print_config_counts(config):
    names = [
        'n_seeds',
        'n_environments',
        'n_agents',
        'n_experiments',
        'n_steps',
    ]
    values = [
        config.n_seeds,
        len(config.environments),
        len(config.agents),
        len(config.experiments),
        config.n_steps,
    ]
    print(
        ', '.join(
            '{} = {}'.format(name, value) for name, value in zip(names, values)
        )
    )


def print_probabilities(config):
    assert len(config.environments) == 1, 'More than one environments exist'
    probabilities = config.environments['env']().probs.tolist()
    print(probabilities)


def collate_data(params_df, results_df):
    print(f'{results_df.shape=}')

    df = pd.merge(results_df, params_df, on="unique_id")
    plt_df = (
        df.groupby(["agent", "t"])
        .agg(
            instant_regret=("instant_regret", np.mean),
            n_instance_regret=("instant_regret", np.size),
        )
        .reset_index()
    )

    print(f'{plt_df.shape=}')
    print(plt_df.head())


def main():
    N_JOBS = 4
    config = config_simple.get_config()

    print_config_counts(config)

    results = []
    for job_id in range(N_JOBS):
        # Running the experiment.
        job_config = config_lib.get_job_config(config, job_id)
        experiment = job_config["experiment"]
        print(experiment)
        experiment.run_experiment()
        experiment_df = pd.DataFrame(experiment.results)
        results.append(experiment_df)

    params_df = config_lib.get_params_df(config)
    results_df = pd.concat(results)

    collate_data(params_df, results_df)
    print_probabilities(config)


if __name__ == '__main__':
    main()
