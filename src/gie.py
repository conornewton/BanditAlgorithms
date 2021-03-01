import math
import random

import numpy as np

from ucb_node import UCBNode
from kl_node import KLNode
from arms import BenoulliArms

class GosInE:
    def __init__(self, n, arms : BenoulliArms, node_type = "UCB", eps = 1, alpha = 1):
        """docstring for __init__"""
        self.arms = arms

        k = arms.no_arms
        sss = math.ceil(k/n) # Sticky set size

        if node_type == "UCB":
            self.nodes = [UCBNode([(((i - 1) * sss + j) % k) for j in range(sss)],
                                  (i * sss) % k,
                                  (i * sss + 1) % k, alpha)
                          for i in range(1, n + 1)]
        else:
            self.nodes = [KLNode([(((i - 1) * sss + j) % k) for j in range(sss)],
                                  (i * sss) % k,
                                  (i * sss + 1) % k)
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


if __name__ == '__main__':
    t = 10000
    arms = BenoulliArms([0.9, 0.5, 0.5, 0.5])
    unif = [[np.random.uniform() for _ in range(t)],
            [np.random.uniform() for _ in range(t)],
            [np.random.uniform() for _ in range(t)],
            [np.random.uniform() for _ in range(t)]]

    print("UCB results")
    ucb = GosInE(2, arms, node_type="UCB")
    ucb.play_unif(t, range(t), unif)

    for node in ucb.nodes:
        print(max(ucb.arms.means) * t- sum(node.rewards))


    print("KLUCB results")
    klucb = GosInE(2, arms, node_type="KL_UCB")
    klucb.play_unif(t, range(t), unif)

    for node in klucb.nodes:
        print(max(klucb.arms.means) * t - sum(node.rewards))
        print(node.times_played)
