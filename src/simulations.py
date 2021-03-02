import numpy as np

import matplotlib.pyplot as plt

from arms import BenoulliArms
from gie import GosInE

t = 10000
# delta = 0.4 # Difference between first and second largest mean rewards

iters = 1

average_regret_kl = np.zeros(t)
average_regret_ucb = np.zeros(t)

## When KLUCB get reward of 0, it never tries that arm again
for i in range(iters):

    arms = BenoulliArms([0.7, 0.5, 0.5, 0.5])
    max_mean = max(arms.means)
    unif = [[np.random.uniform() for _ in range(t)],
            [np.random.uniform() for _ in range(t)],
            [np.random.uniform() for _ in range(t)],
            [np.random.uniform() for _ in range(t)]]

    # print("UCB results")
    ucb = GosInE(2, arms, node_type="UCB")
    ucb.play_unif(t, range(t), unif)

    average_regret_ucb = np.add(average_regret_ucb, np.true_divide(np.add(ucb.nodes[0].regret(max_mean), ucb.nodes[1].regret(max_mean)), 2))
    # for node in ucb.nodes:
    #     print(max(ucb.arms.means) * t- sum(node.rewards))
    #     print(node.times_played)

    # print("KLUCB results")
    klucb = GosInE(2, arms, node_type="KL_UCB")
    klucb.play_unif(t, range(t), unif)

    # for node in klucb.nodes:
    #     print(max(klucb.arms.means) * t - sum(node.rewards))
    #     print(node.times_played)
    #     print(node.empirical_means)

    average_regret_kl = np.add(average_regret_kl, np.true_divide(np.add(klucb.nodes[0].regret(max_mean), klucb.nodes[1].regret(max_mean)), 2))

average_regret_kl = np.true_divide(average_regret_kl, iters)
average_regret_ucb = np.true_divide(average_regret_ucb, iters)

plt.plot(average_regret_ucb, 'r')
plt.plot(average_regret_kl, 'g')
plt.xlabel("T")
plt.ylabel("Regret")
plt.show()
