import math

from node import Node

class UCBNode(Node):
    def __init__(self, sticky_arms, arm1, arm2, alpha):
        Node.__init__(self, sticky_arms, arm1, arm2)
        self.alpha = alpha


    def play(self, t):
        max_arm_id = 0
        ucb = [0] * len(self.arms)

        for i in range(len(ucb)):
            if self.times_played[self.arms[i]] == 0:
                return self.arms[i]

            ucb[i] = self.empirical_means[self.arms[i]] + math.sqrt(self.alpha * math.log(t) / self.times_played[self.arms[i]])
            if ucb[i] > ucb[max_arm_id]:
                max_arm_id = i

        return self.arms[max_arm_id]
