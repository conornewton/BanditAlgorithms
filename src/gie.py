import math
import random

import numpy as np

from ucb_node import UCBNode
from kl_node import KLNode
from arms import BenoulliArms

class GosInE:
    def __init__(self, n, arms, node_type = "UCB", eps = 1, alpha = 1):
        """docstring for __init__"""
        self.arms = arms

        k = arms.no_arms
        sss = math.ceil(k/n) # Sticky set size

        if node_type == "UCB":
            self.nodes = [UCBNode([(((i - 1) * sss + j) % k) for j in range(sss)],
                                  (i * sss) % k,
                                  (i * sss + 1) % k, alpha, k)
                          for i in range(1, n + 1)]
        else:
            self.nodes = [KLNode([(((i - 1) * sss + j) % k) for j in range(sss)],
                                  (i * sss) % k,
                                  (i * sss + 1) % k, k)
                          for i in range(1, n + 1)]
        self.time = 0
        self.phase = 0

        self.eps = eps
        self.alpha = alpha

        self.comm_matrix = []
        for i in range(n):
            row_i = []
            prob = 1 / (n - 1)
            for j in range(n):
                if j == i:
                    row_i.append(0)
                else:
                    row_i.append(prob)
            self.comm_matrix.append(row_i)

    def play(self, t, comm_budget):
        comm_rounds = self.adjust_comm_budget(comm_budget)

        for i in range(t):
            for node in self.nodes:
                arm_id = node.play(i)
                node.recieve_reward(arm_id, self.arms.play(arm_id))

            if i == comm_rounds[self.phase]:
                for j in range(len(self.nodes)):
                    comm_node_id = random.choices(range(len(self.nodes)), self.comm_matrix[j], k = 1)[0]
                    self.nodes[j].recieve_recommendation(self.nodes[comm_node_id].give_recommendation())

                self.phase += 1

    # Takes a list or uniform randomly generated numbers used for the random generation
    def play_unif(self, t, comm_budget, unif):
        comm_rounds = self.adjust_comm_budget(comm_budget)

        for i in range(t):
            for node in self.nodes:
                arm_id = node.play(i)
                node.recieve_reward(arm_id, self.arms.play_unif(arm_id, unif[arm_id][i]))

            if i == comm_rounds[self.phase]:
                for j in range(len(self.nodes)):
                    comm_node_id = random.choices(range(len(self.nodes)), self.comm_matrix[j], k = 1)[0]
                    self.nodes[j].recieve_recommendation(self.nodes[comm_node_id].give_recommendation())

                self.phase += 1

    def adjust_comm_budget(self, comm_budget):
        return [max(next(filter(lambda t: t >= i, comm_budget)), math.ceil((1 + i) ** (1 + self.eps))) for i in range(len(comm_budget))]

    def average_regret(self):
        avg_regret = np.zeros(len(self.nodes[0].rewards))

        for node in self.nodes:
            avg_regret = np.add(avg_regret, node.regret(self.arms.max_mean()))

        avg_regret = np.true_divide(avg_regret, len(self.nodes))
        return avg_regret
