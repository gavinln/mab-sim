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
import time

from multiprocessing import Pool

from base import config_lib
from finite_arm import config_simple

import numpy as np
import pandas as pd
import plotnine as gg


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


def collate_instant_regret(params_df, results_df):
    'combine data using pandas'
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
    return plt_df


def save_plot_instant_regret(plt_df, probabilities):
    title = 'probabilities_{0[0]}_{0[1]}'.format(
        [prob * 100 for prob in probabilities]
    )

    # Plotting and analysis (uses plotnine by default)
    gg.theme_set(gg.theme_bw(base_size=16, base_family="serif"))
    gg.theme_update(figure_size=(12, 8))

    p = (
        gg.ggplot(plt_df)
        + gg.aes("t", "instant_regret", colour="agent")
        + gg.geom_line()
        + gg.ggtitle(title)
    )
    p.save(filename='{}.png'.format(title), verbose=True)
    # print(p)


def collate_cum_regret(params_df, results_df):
    'combine data using pandas'
    print(f'{results_df.shape=}')
    plt_df = pd.merge(results_df, params_df, on="unique_id")
    print(f'{plt_df.shape=}')
    return plt_df


def save_plot_cum_regret(plt_df, probabilities):
    title = 'cum_regret_probabilities_{0[0]}_{0[1]}'.format(
        [prob * 100 for prob in probabilities]
    )

    # Plotting and analysis (uses plotnine by default)
    gg.theme_set(gg.theme_bw(base_size=16, base_family="serif"))
    gg.theme_update(figure_size=(12, 8))

    p = (
        gg.ggplot(plt_df)
        + gg.geom_boxplot(gg.aes(x='agent', y='cum_regret'))
        + gg.ggtitle(title)
    )
    p.save(filename='{}.png'.format(title), verbose=True)
    # print(p)


def run_experiment(experiment):
    experiment.run_experiment()
    return pd.DataFrame(experiment.results)


def run_experiments_serial(experiments):
    return [run_experiment(experiment) for experiment in experiments]


def run_experiments_parallel(experiments):
    results = []
    with Pool(14) as p:
        results = p.map(run_experiment, experiments)
    return results


def main():
    parallel = True

    assert parallel in (True, False)

    N_JOBS = 100
    print(f'{N_JOBS=}')

    # probs = [0.2, 0.3]
    # probs = [0.0022, 0.0024, 0.0026, 0.0028, 0.0030]

    # probs = [0.6, 0.9]
    # probs = [0.02, 0.03]
    probs = [0.002, 0.003]
    n_steps = 10000
    n_seeds = 10000
    config = config_simple.get_config(probs, n_steps, n_seeds)

    print_config_counts(config)

    experiments = []
    for job_id in range(N_JOBS):
        job_config = config_lib.get_job_config(config, job_id)
        experiment = job_config["experiment"]
        experiments.append(experiment)

    start_time = time.time()
    if parallel:
        results = run_experiments_parallel(experiments)
    else:
        results = run_experiments_serial(experiments)

    elapsed_time = time.time() - start_time
    print('elasped_time {:.0f} seconds'.format(elapsed_time))

    results_df = pd.concat(results)
    params_df = config_lib.get_params_df(config)

    # plt_df = collate_instant_regret(params_df, results_df)
    plt_df = collate_cum_regret(params_df, results_df)
    print(plt_df)

    assert len(config.environments) == 1, 'More than one environments exist'
    probabilities = config.environments['env']().probs.tolist()

    # save_plot_instant_regret(plt_df, probabilities)
    save_plot_cum_regret(plt_df, probabilities)


if __name__ == '__main__':
    main()
