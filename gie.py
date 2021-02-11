import math
import random
import numpy as np

from scipy.stats import bernoulli

# TODO: break ties randomly

# TODO: implement communication
# TODO: random sticky sets
# TODO: refactor
# TODO: unit tests

class SyncGosInE:
    def __init__(self, n, k, eps = 1, alpha = 1):
        # choose the means of the arms randomly
        self.arms_means = [np.random.uniform() for _ in range(k)]
        # self.arms_means.sort(reverse=True)

        sss = math.ceil(k/n) # Sticky set size
        self.nodes = [SyncGIENode([(((i - 1) * sss + j) % k) for j in range(sss)],
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
                arm_id = node.play(i, self.alpha)
                node.recieve_reward(arm_id, bernoulli.rvs(self.arms_means[arm_id], size = 1))

            if i == comm_rounds[self.phase]:
                for j in range(len(self.nodes)):
                    comm_node_id = random.choices(range(len(self.nodes)), self.comm_matrix[j], k = 1)[0]
                    self.nodes[j].recieve_recommendation(self.nodes[comm_node_id].give_recommendation())

                self.phase += 1

    def adjust_comm_budget(self, comm_budget):
        return [max(next(filter(lambda t: t >= i, comm_budget)), math.ceil((1 + i) ** (1 + self.eps))) for i in range(len(comm_budget))]

    def __str__(self):
        return str(self.arms_means)

class SyncGIENode:
    def __init__(self, sticky_arms, arm1, arm2):
        self.arms = sticky_arms
        self.arms.append(arm1)
        self.arms.append(arm2)

        self.history = [] # Previous arm pulls
        self.rewards = [] # Rewards from previous arm pulls

        self.rewards_per_arm = [0] * 1000
        self.times_played = [0] * 1000 # Times each arm has been played
        self.empirical_means = [0] * 1000

        self.phase = 0
        self.times_played_phase = [0] * len(self.arms)

    def play(self, t, alpha):
        # Each arm should be played at least once
        if t < len(self.arms):
            return self.arms[t]

        max_arm_id = 0
        ucb = [0] * len(self.arms)

        for i in range(len(ucb)):
            if self.times_played[self.arms[i]] == 0:
                return self.arms[i]

            ucb[i] = self.empirical_means[self.arms[i]] + math.sqrt(alpha * math.log(t) / self.times_played[self.arms[i]])
            if ucb[i] > ucb[max_arm_id]:
                max_arm_id = i

        return self.arms[max_arm_id]

    def recieve_reward(self, arm_id, reward):
        self.history.append(arm_id)
        self.rewards.append(reward)

        self.rewards_per_arm[arm_id] += reward
        self.times_played[arm_id] += 1
        self.empirical_means[arm_id] = self.rewards_per_arm[arm_id] / self.times_played[arm_id]

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

    def regret(self):
        pass

if __name__ == '__main__':
    gie = SyncGosInE(n = 10, k = 100)

    gie.play(100000, range(1,100000))

    for node in gie.nodes:
        print(max(gie.arms_means) * 100000 - sum(node.rewards))
