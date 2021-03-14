from scipy.stats import bernoulli
from random import shuffle

class BenoulliArms:
    def __init__(self, arm_means):
        self.means   = arm_means
        self.no_arms = len(arm_means)

    def max_mean(self):
        return max(self.means)

    def best_arm(self):
        """
            Returns the id (index) of the arm with the highest mean
        """
        return self.means.index(self.max_mean())

    def play(self, arm_id):
        return bernoulli.rvs(self.means[arm_id], size = 1)

    def play_unif(self, arm_id, unif):
        if unif <= self.means[arm_id]:
            return 1
        else:
            return 0


def param_arms(delta, high, low, k):
    """
    Simulates the Multi-agent MAB for a range of parameters

        Parameters:
            delta (float): difference between the largest and second largest mean
            high  (float): highest mean
            low   (float): lowest mean
        Returns:
            arms (BernoulliArms): An object of BernoulliArms
    """
    arm_means = []
    arm_means.append(high)

    width = (high - delta - low) / (k - 1)

    for i in range(k - 1):
        arm_means.append(high - delta - width * i)

    shuffle(arm_means)

    return BenoulliArms(arm_means)

