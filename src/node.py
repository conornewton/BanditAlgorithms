import math
import random
import numpy as np

from scipy.optimize import newton

class Node:
    def __init__(self, sticky_arms, arm1, arm2):
        self.arms = sticky_arms
        self.arms.append(arm1)
        self.arms.append(arm2)

        self.history = [] # Previous arm pulls
        self.rewards = [] # Rewards from previous arm pulls

        self.recommendations_recieved = []

        self.rewards_per_arm = [0] * len(self.arms)
        self.times_played = [0] * len(self.arms)# Times each arm has been played
        self.empirical_means = [0] * len(self.arms)

        self.phase = 0
        self.times_played_phase = [0] * len(self.arms)

    def recieve_reward(self, arm_id, reward):
        self.history.append(arm_id)
        self.rewards.append(reward)

        self.rewards_per_arm[arm_id] += reward
        self.times_played[arm_id] += 1
        self.empirical_means[arm_id] = self.rewards_per_arm[arm_id] / self.times_played[arm_id]

        self.recommendations_recieved.append(arm_id)

    def give_recommendation(self):
        return self.arms[self.times_played_phase.index(max(self.times_played_phase))]

    def recieve_recommendation(self, arm_id):
        if arm_id in (self.arms[-1], self.arms[-2]):
            pass
        elif self.times_played_phase[-1] > self.times_played_phase[-2]:
            self.arms[-2] = arm_id
        else:
            self.arms[-1] = arm_id

        # Move to next phase
        self.phase += 1
        self.times_played_phase = [0] * len(self.arms)

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

