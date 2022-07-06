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

import functools
import collections

from base import config_lib
from finite_arm import config_simple

import numpy as np
import pandas as pd
import plotnine as gg

from base.experiment import BaseExperiment
from finite_arm.agent_finite import FiniteBernoulliBanditEpsilonGreedy
from finite_arm.agent_finite import FiniteBernoulliBanditTS
from finite_arm.env_finite import FiniteArmedBernoulliBandit


assert False, "may not be used"


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


class Experiment:
    """Simple experiment that logs regret and action taken."""

    def __init__(
        self, agent, environment, n_steps, seed=0, rec_freq=1, unique_id="NULL"
    ):
        """Setting up the experiment.

        Note that unique_id should be used to identify the job later for
        analysis.

        """
        self.agent = agent
        self.environment = environment
        self.n_steps = n_steps
        self.seed = seed
        self.unique_id = unique_id

        self.results = []
        self.rec_freq = rec_freq

    def run_step_maybe_log(self, t):
        # Evolve the bandit (potentially contextual) for one step and
        # pick action
        observation = self.environment.get_observation()
        action = self.agent.pick_action(observation)

        # Compute useful stuff for regret calculations
        optimal_reward = self.environment.get_optimal_reward()
        expected_reward = self.environment.get_expected_reward(action)
        reward = self.environment.get_stochastic_reward(action)

        # Update the agent using realized rewards + bandit learing
        self.agent.update_observation(observation, action, reward)

        # Log whatever we need for the plots we will want to use.
        instant_regret = optimal_reward - expected_reward
        self.cum_regret += instant_regret

        # Advance the environment (used in nonstationary experiment)
        self.environment.advance(action, reward)

        if (t + 1) % self.rec_freq == 0:
            data_dict = {
                "t": (t + 1),
                "instant_regret": instant_regret,
                "cum_regret": self.cum_regret,
                "action": action,
                "unique_id": self.unique_id,
                "agent": self.agent.name,
                "seed": self.seed,
            }
            self.results.append(data_dict)

    def run_experiment(self):
        """Run the experiment for n_steps and collect data."""
        np.random.seed(self.seed)
        self.cum_regret = 0
        self.cum_optimal = 0

        for t in range(self.n_steps):
            self.run_step_maybe_log(t)

    def __str__(self):
        names = 'n_steps seed unique_id, agent, environment'.split(' ')
        values = [
            self.n_steps,
            self.seed,
            self.unique_id,
            self.agent.__class__.__name__,
            self.environment.__class__.__name__,
        ]
        return ', '.join(
            f'{name}: {value}' for name, value in zip(names, values)
        )


def get_config(probs: list[float], n_steps: int, n_seeds: int, rec_freq: int):
    """Generates the config for the experiment."""

    assert rec_freq >= 1 and rec_freq <= n_steps

    # check all probs between 0 and 1:p
    assert all(
        0 <= prob and prob <= 1 for prob in probs
    ), 'Not all probabilities are between 0 and 1'

    # check all pro
    n_arm = len(probs)

    agents = {
        "greedy": functools.partial(FiniteBernoulliBanditEpsilonGreedy, n_arm),
        "ts": functools.partial(FiniteBernoulliBanditTS, n_arm),
    }
    environments = {
        "env": functools.partial(FiniteArmedBernoulliBandit, probs)
    }
    experiments = {"finite_simple": Experiment}

    exp_list = []
    unique_id = 0
    for seed in range(n_seeds):
        for env_name, EnvC in environments.items():
            for agent_name, AgentC in agents.items():
                for exp_name, ExperimentC in experiments.items():
                    print(f'{seed=}, {env_name=}, {agent_name=}, {exp_name=}')
                    exp = ExperimentC(
                        AgentC(agent_name),
                        EnvC(),
                        n_steps,
                        seed=seed,
                        rec_freq=rec_freq,
                        unique_id=unique_id,
                    )
                    exp_list.append(exp)
                    unique_id += 1
    return exp_list


def main():
    assert False, "May not be used"

    N_JOBS = 4

    probs = [0.2, 0.3]
    # n_steps = 10000
    # n_seeds = 10000
    n_steps = 1000
    n_seeds = 2
    rec_freq = 100

    exp_list = get_config(probs, n_steps, n_seeds, rec_freq)

    results = []
    for experiment in exp_list:
        experiment.run_experiment()
        experiment_df = pd.DataFrame(experiment.results)
        results.append(experiment_df)

    results_df = pd.concat(results)
    print(results_df)

    config = config_simple.get_config(probs, n_steps, n_seeds)

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

    results_df = pd.concat(results)
    print(results_df)

    params_df = config_lib.get_params_df(config)
    print(params_df)

    df = pd.merge(results_df, params_df, on="unique_id")
    print(df)

    # collate_data(params_df, results_df)
    # print_probabilities(config)


if __name__ == '__main__':
    main()
