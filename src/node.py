import math
import random
import numpy as np

from scipy.optimize import newton

class Node:
    def __init__(self, sticky_arms, arm1, arm2, k, alpha):
        self.arms = sticky_arms
        self.arms.append(arm1)
        self.arms.append(arm2)

        self.history = [] # Previous arm pulls
        self.rewards = [] # Rewards from previous arm pulls

        self.recommendations_recieved = []
        self.recommendations_recieved_node = []

        self.rewards_per_arm = [0] * k
        self.times_played = [0] * k
        self.empirical_means = [0] * k
        self.phase = 0
        self.times_played_phase = [0] * k

        self.alpha = alpha

    def recieve_reward(self, arm_id, reward):
        self.history.append(arm_id)
        self.rewards.append(reward)
        self.rewards_per_arm[arm_id] += reward
        self.times_played[arm_id] += 1
        self.times_played_phase[arm_id] += 1
        self.empirical_means[arm_id] = self.rewards_per_arm[arm_id] / self.times_played[arm_id]


    def give_recommendation(self):
        return self.times_played_phase.index(max(self.times_played_phase))

    def recieve_recommendation(self, arm_id, comm_node):
        self.recommendations_recieved.append(arm_id)
        self.recommendations_recieved_node.append(comm_node)
        if arm_id in self.arms:
            pass
        elif self.times_played_phase[self.arms[-1]] > self.times_played_phase[self.arms[-2]]:
            self.arms[-2] = arm_id
        else:
            self.arms[-1] = arm_id


    def next_phase(self):
        # Move to next phase
        self.phase += 1
        self.times_played_phase = [0] * len(self.times_played_phase)

    def cum_reward(self):
        cum = self.rewards
        for i in range(1, len(cum)):
            cum[i] += cum [i-1]

        return cum

    def regret(self, mean):
        cum = self.cum_reward()
        for i in range(len(cum)):
            cum[i] = i * mean - cum[i]
        return cum


    def when_recieved_arm(self, arm_id):
        """
            Returns the phase when an arm was recommended
        """
        if arm_id in self.arms[:-2]:
            return 0
        return self.recommendations_recieved.index(arm_id)
