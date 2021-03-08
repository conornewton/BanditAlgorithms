import math
import numpy as np

from scipy.optimize import bisect

from node import Node

class KLNode(Node):
    def __init__(self, sticky_arms, arm1, arm2, k, c = 0):
        Node.__init__(self, sticky_arms, arm1, arm2, k)
        self.c = c


    def play(self, t):
        max_arm_id = 0
        kl_ucb = [0] * len(self.arms)

        # If an arm has not been played, play it!
        for i in range(len(kl_ucb)):
            if self.times_played[self.arms[i]] == 0:
                return self.arms[i]


        for i in range(len(kl_ucb)):
            empirical_mean = self.empirical_means[self.arms[i]]
            times_played = self.times_played[self.arms[i]]
            kl_ucb_ineq = lambda x: times_played * self.KL(empirical_mean, x) - self.exploration(t)

            if kl_ucb_ineq(1) <= 0:
                kl_ucb[i] = 1
            else:
                try:
                    kl_ucb[i] = bisect(kl_ucb_ineq, empirical_mean, 1)
                except ValueError as e:
                    print(empirical_mean)
                    print(kl_ucb_ineq(empirical_mean), kl_ucb_ineq(1))


            # kl_ucb[i] = self.max_ineq(kl_ucb_ineq, empirical_mean)
            # print(kl_ucb)

            if kl_ucb[i] > kl_ucb[max_arm_id]:
                max_arm_id = i

        return self.arms[max_arm_id]

    def exploration(self, t):
        return math.log(t) + self.c * math.log(math.log(t))


    def KL(self, p, q):
        try:
            if p == 0 and q != 1:
                return math.log(1 / (1 - q))
            if p == 1 and q != 0:
                return math.log(1 / q)
            if q == 0:
                return 10000000000000000 # Infinity
            if q == 1:
                return 10000000000000000 # Infinity
        # try:
            return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
        except Exception as e:
            return 10000000000000000

    def max_ineq(self, ineq, emean, iters=1000):
        x0 = emean
        x1 = 1

        if ineq(1) <= 0:
            return 1

        for _ in range(iters):
            x_new = 0.5 * (x0 + x1)
            if ineq(x_new) <= 0:
                x0 = x_new
            else:
                x1 = x_new

        return x0

