import numpy as np
import pymc3 as pm


class MultiArmedBandit2(object):
    """
    A Multi-armed Bandit
    """

    def __init__(self, k):
        self.k = k
        self.action_values = np.zeros(k)
        self.optimal = 0

    def reset(self):
        self.action_values = np.zeros(self.k)
        self.optimal = 0

    def pull(self, action):
        return 0, True


class GaussianBandit2(MultiArmedBandit2):
    """
    Gaussian bandits model the reward of a given arm as normal distribution
    with provided mean and standard deviation.
    """

    def __init__(self, k, mu=0, sigma=1):
        super(GaussianBandit2, self).__init__(k)
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.action_values = np.random.normal(self.mu, self.sigma, self.k)
        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        return (
            np.random.normal(self.action_values[action]),
            action == self.optimal,
        )


class BinomialBandit2(MultiArmedBandit2):
    """
    The Binomial distribution models the probability of an event occurring with
    p probability k times over N trials i.e. get heads on a p-coin k times on
    N flips.

    In the bandit scenario, this can be used to approximate a discrete user
    rating or "strength" of response to a single event.
    """

    def __init__(self, k, n, p, t):
        super(BinomialBandit2, self).__init__(k)
        self.n = n
        self.p = p
        self.t = t
        self.model = pm.Model()
        with self.model:
            self.bin = pm.Binomial(
                'binomial',
                n=n * np.ones(k, dtype=np.int),
                p=np.ones(k) / n,
                shape=(1, k),
                transform=None,
            )
        self._samples = None
        self._cursor = 0

        self.reset()

    def reset(self):
        self.action_values = self.p
        self.bin.distribution.p = self.action_values
        if self.t is not None:
            self._samples = self.bin.random(size=self.t).squeeze()
            self._cursor = 0

        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        reward, is_optimal = self.sample[action], action == self.optimal
        return reward, is_optimal

    @property
    def sample(self):
        val = self._samples[self._cursor]
        self._cursor += 1
        return val


class BernoulliBandit2(BinomialBandit2):
    """
    The Bernoulli distribution models the probability of a single event
    occurring with p probability i.e. get heads on a single p-coin flip. This
    is the special case of the Binomial distribution where N=1.

    In the bandit scenario, this can be used to approximate a hit or miss
    event, such as if a user clicks on a headline, ad, or recommended product.
    """

    def __init__(self, k, p, t):
        super(BernoulliBandit2, self).__init__(k, 1, p=p, t=t)


class MultiArmedBandit(object):
    """
    A Multi-armed Bandit
    """

    def __init__(self, arm_count: int):
        assert arm_count > 0, 'Number of bandit arms should be > 0'
        self.arm_count = arm_count  # number of arms

    def reset(self):
        pass

    def pull(self, arm: int):
        return 0, True


class BinomialBandit(MultiArmedBandit):
    """
    The Binomial distribution models the probability of an event occurring with
    p probability k times over N trials i.e. get heads on a p-coin k times on
    N flips.

    In the bandit scenario, this can be used to approximate a discrete user
    rating or "strength" of response to a single event.
    """

    def __init__(self, arm_count: int, n: int, p: np.ndarray, step_count: int):
        super(BinomialBandit, self).__init__(arm_count)

        assert n >= 0
        assert (
            arm_count == p.size
        ), 'Number of probabilities should equal number of arms'
        assert step_count > 0, 'Number of bandit steps should be > 0'

        self.n = n  # n trials of a binomial distribution
        self.p = p  # p probability of success in each trial
        self.step_count = step_count  # number of bandit draws/steps
        self._sample: np.ndarray
        self._current_step = 0

        self.reset()

    def reset(self):
        rng = np.random.default_rng()
        self._samples = rng.binomial(
            self.n, self.p, (self.step_count, self.p.size)
        )
        self._current_step = 0

    def is_optimal_arm(self, arm: int, step: int):
        assert arm < self.arm_count
        assert step < self.step_count

        step_values = self._samples[step, :]
        max_value = np.max(step_values)
        max_index = np.nonzero(step_values >= max_value)[0]
        is_optimal = np.any(max_index == arm)
        return is_optimal

    def pull(self, arm: int):
        assert arm >= 0 and arm < self.arm_count
        assert self._current_step < self.step_count
        reward = self._samples[self._current_step, arm]
        is_optimal = self.is_optimal_arm(arm, self._current_step)
        self._current_step += 1
        return reward, is_optimal


class BernoulliBandit(BinomialBandit):
    """
    The Bernoulli distribution models the probability of a single event
    occurring with p probability i.e. get heads on a single p-coin flip. This
    is the special case of the Binomial distribution where N=1.

    In the bandit scenario, this can be used to approximate a hit or miss
    event, such as if a user clicks on a headline, ad, or recommended product.
    """

    def __init__(self, arm_count, p, step_count):
        super(BernoulliBandit, self).__init__(
            arm_count, 1, p=p, step_count=step_count
        )
