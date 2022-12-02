import numpy as np

class OUNoise(object):
    """Ornstein-Ulhenbeck Process Noise to Improve Explore-Exploit Process"""

    def __init__(self, action_space, mean=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        # Hyperparameters
        self.mean = mean
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()


    def reset(self):
        self.state = np.ones(self.action_dim) * self.mean


    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mean - x) + self.sigma * \
            np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - \
            (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)

        return np.clip(action + ou_state, self.low, self.high)