import numpy as np

import matplotlib.pyplot as plt

from arms import param_arms
from gie import GosInE
from simulation_util import *

t = 10000
arms = param_arms(0.2, 0.7, 0.5, 4)

out_ucb = GosInE(2, arms, node_type = "UCB", eps = 1, alpha = 1)
out_ucb.play(t, range(t))
average_regret_ucb = average_regret(out_ucb.nodes, 0.7)

out_klucb = GosInE(2, arms, node_type = "KL-UCB", eps = 1, alpha = 1)
out_klucb.play(t, range(t))
average_regret_klucb = average_regret(out_klucb.nodes, 0.7)


plt.plot(average_regret_ucb, 'r')
plt.plot(average_regret_klucb, 'g')
plt.xlabel("T")
plt.ylabel("Regret")
plt.show()
