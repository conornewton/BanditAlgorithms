from scipy.stats import bernoulli

class BenoulliArms:
    def __init__(self, arm_means):
        self.means   = arm_means
        self.no_arms = len(arm_means)

    def play(self, arm_id):
        return bernoulli.rvs(self.means[arm_id], size = 1)

    def play_unif(self, arm_id, unif):
        if unif <= self.means[arm_id]:
            return 1
        else:
            return 0
