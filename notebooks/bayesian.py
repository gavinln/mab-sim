"""
Takes advantage of multicore systems to speed up the simulation runs.
"""
import numpy as np
import sys

# import matplotlib
# matplotlib.use('qt4agg')
# from bandits.agent import Agent, BetaAgent
from bandits.bandit import BernoulliBandit
from bandits.policy import RandomPolicy
from bandits.policy import GreedyPolicy
from bandits.policy import EpsilonGreedyPolicy
from bandits.agent import Agent

# from bandits.policy import GreedyPolicy, EpsilonGreedyPolicy, UCBPolicy
# from bandits.environment import Environment


# class BernoulliExample:
#     label = 'Bayesian Bandits - Bernoulli'
#     bandit = BernoulliBandit(10, p=0.5, t=3*1000)
#     agents = [
#         Agent(bandit, EpsilonGreedyPolicy(0.1)),
#         Agent(bandit, UCBPolicy(1)),
#         BetaAgent(bandit, GreedyPolicy())
#     ]


# class BinomialExample:
#     label = 'Bayesian Bandits - Binomial (n=5)'
#     bandit = BinomialBandit(10, n=5, p=0.5, t=3*1000)
#     agents = [
#         Agent(bandit, EpsilonGreedyPolicy(0.1)),
#         Agent(bandit, UCBPolicy(1)),
#         BetaAgent(bandit, GreedyPolicy())
#     ]


if __name__ == '__main__':
    experiments = 1
    total_steps = 1_000_000

    bandit = BernoulliBandit(3, np.array([0.7, 0.1, 0.5]), total_steps)
    # agent = Agent(bandit, RandomPolicy())
    agent = Agent(bandit, GreedyPolicy())
    total_optimal = 0
    for step in range(total_steps):
        arm, reward, is_optimal = agent.choose()
        # print(step, arm, reward, is_optimal)
        total_optimal += is_optimal
    print('Optimal fraction: {}'.format(total_optimal/total_steps))

    # example = BernoulliExample()
    # example = BinomialExample()

    # env = Environment(example.bandit, example.agents, example.label)
    # scores, optimal = env.run(trials, experiments)
    # env.plot_results(scores, optimal)
    # env.plot_beliefs()
