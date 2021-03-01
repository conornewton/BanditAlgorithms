import math

import numpy as np

from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from node import Node

class KLNode(Node):
    def __init__(self, sticky_arms, arm1, arm2):
        Node.__init__(self, sticky_arms, arm1, arm2)


    def play(self, t):
        max_arm_id = 0
        kl_ucb = [0] * len(self.arms)

        for i in range(len(kl_ucb)):
            if self.times_played[self.arms[i]] == 0:
                return self.arms[i]


            empirical_mean = self.empirical_means[self.arms[i]]
            times_played = self.times_played[self.arms[i]]
            kl_ucb_ineq = lambda x: times_played * self.KL(empirical_mean, x) - math.log(t)

            # nlc1 = NonlinearConstraint(kl_ucb_ineq, np.inf, 0)
            # nlc2 = NonlinearConstraint(lambda x: x, 0, 1)
            # # bounds = Bounds(0.000001, 0.999999, keep_feasible = True)

            # res = minimize(lambda x: -x, 0.5, constraints=[nlc1, nlc2], method="COBYLA")
            # kl_ucb[i] = res.x
            # # minimize(lambda x: -x, 0.5, bounds = bounds, constraints=[nlc1, nlc2], method="COBYLA")
            # print(res.success)

            kl_ucb[i] = self.max_ineq(kl_ucb_ineq, empirical_mean)
            # print(kl_ucb)

            if kl_ucb[i] > kl_ucb[max_arm_id]:
                max_arm_id = i

        return self.arms[max_arm_id]


    def KL(self, p, q):
        try:
            if q <= 0 :
                return 1000000000000000
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

