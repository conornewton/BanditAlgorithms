import math

from node import Node

class UCBNode(Node):
    def __init__(self, sticky_arms, arm1, arm2, k, alpha = 4):
        Node.__init__(self, sticky_arms, arm1, arm2, k, alpha)

    def play(self, t):
        max_arm_id = 0
        ucb = [0] * len(self.arms)

        for i in range(len(ucb)):
            if self.times_played[self.arms[i]] == 0:
                return self.arms[i]

            ucb[i] = self.empirical_means[self.arms[i]] + math.sqrt(self.exploration(t) / (2 * self.times_played[self.arms[i]]))

            if ucb[i] > ucb[max_arm_id]:
                max_arm_id = i

        return self.arms[max_arm_id]

    def exploration(self, t):
        return math.log(1 + (t ** self.alpha) * math.log(t) ** 2) # From bandit algorithms
