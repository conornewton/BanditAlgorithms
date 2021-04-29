import math
import random

import numpy as np

from ucb_node import UCBNode
from kl_node import KLNode
from arms import BenoulliArms
from gossip_matrix import *

class GosInE2:
    def __init__(self, n, arms, node_type = "UCB", gossip_matrix = "COMPLETE", alpha = 1):
        """docstring for __init__"""
        self.arms = arms

        k = arms.no_arms
        sss = math.ceil(k/n) # Sticky set size

        if node_type == "UCB":
            self.nodes = [UCBNode([(((i - 1) * sss + j) % k) for j in range(sss)],
                                  (i * sss) % k,
                                  (i * sss + 1) % k, k, alpha)
                          for i in range(1, n + 1)]
        else:
            self.nodes = [KLNode([(((i - 1) * sss + j) % k) for j in range(sss)],
                                  (i * sss) % k,
                                  (i * sss + 1) % k, k, alpha)
                          for i in range(1, n + 1)]

        self.time = 0
        self.phase = 0

        if gossip_matrix == "COMPLETE":
            self.gossip_matrix = CompleteGossipMatrix(n)
        elif gossip_matrix == "STAR":
            self.gossip_matrix = StarGossipMatrix(n)
        elif gossip_matrix == "RING":
            self.gossip_matrix = RingGossipMatrix(n)

    def play(self, t, comm_rounds):
        for i in range(t):
            for node in self.nodes:
                arm_id = node.play(i)
                node.recieve_reward(arm_id, self.arms.play(arm_id))

            if i == comm_rounds[self.phase]:
                for j in range(len(self.nodes)):
                    comm_node_id = self.gossip_matrix.sample(j)
                    self.nodes[j].recieve_recommendation2(self.nodes[comm_node_id].give_recommendation(), comm_node_id)

                self.phase += 1

    def play_unif_no_comm(self, t, unif):
        for i in range(t):
            for node in self.nodes:
                arm_id = node.play(i)
                node.recieve_reward(arm_id, self.arms.play_unif(arm_id, unif[arm_id][i]))

    # Takes a list or uniform randomly generated numbers used for the random generation
    def play_unif(self, t, comm_rounds, unif):
        for i in range(t):
            for node in self.nodes:
                arm_id = node.play(i)
                node.recieve_reward(arm_id, self.arms.play_unif(arm_id, unif[arm_id][i]))

            if i == comm_rounds[self.phase]:
                for j in range(len(self.nodes)):
                    comm_node_id = self.gossip_matrix.sample(j)
                    self.nodes[j].recieve_recommendation2(self.nodes[comm_node_id].give_recommendation(), comm_node_id)

                self.phase += 1
                for node in self.nodes:
                    node.next_phase()

    def play_unif_index(self, t, comm_rounds, unif):
        for i in range(t):
            for node in self.nodes:
                arm_id = node.play(i)
                node.recieve_reward(arm_id, self.arms.play_unif(arm_id, unif[arm_id][i]))

            if i == comm_rounds[self.phase]:
                for j in range(len(self.nodes)):
                    comm_node_id = self.gossip_matrix.sample(j)
                    self.nodes[j].recieve_recommendation2(self.nodes[comm_node_id].play(i), comm_node_id)

                self.phase += 1
                for node in self.nodes:
                    node.next_phase()


    def average_regret(self):
        avg_regret = np.zeros(len(self.nodes[0].rewards))

        for node in self.nodes:
            avg_regret = np.add(avg_regret, node.regret(self.arms.max_mean()))

        avg_regret = np.true_divide(avg_regret, len(self.nodes))
        return avg_regret