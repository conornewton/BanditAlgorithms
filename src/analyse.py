import pickle
import numpy as np

from gie import GosInE
from simulation_util import *

ucb0 = pickle.load(open("data/ucb_100000_20_5_0.00_0.75_0.50_0.50_0.p", "rb"))
ucb1 = pickle.load(open("data/ucb_100000_20_5_0.00_0.75_0.50_0.50_1.p", "rb"))

reg0 = ucb0.average_regret()
reg1 = ucb1.average_regret()

res_ucb  = average_results([reg0, reg1])

klucb0 = pickle.load(open("data/klucb_100000_20_5_0.00_0.75_0.50_0.50_0.p", "rb"))
klucb1 = pickle.load(open("data/klucb_100000_20_5_0.00_0.75_0.50_0.50_1.p", "rb"))

klreg0 = klucb0.average_regret()
klreg1 = klucb1.average_regret()

res_klucb = average_results([klreg0, klreg1])

print(ucb0.nodes[0].times_played)
print(ucb0.nodes[1].times_played)
print(ucb0.nodes[2].times_played)
print(ucb0.nodes[3].times_played)
print(ucb0.nodes[4].times_played)

print(klucb0.nodes[0].times_played)
print(klucb0.nodes[1].times_played)
print(klucb0.nodes[2].times_played)
print(klucb0.nodes[3].times_played)
print(klucb0.nodes[4].times_played)

best_arm = ucb0.arms.best_arm()
print(best_arm)


print(ucb0.nodes[0].recommendations_recieved[1:10])
print(ucb0.nodes[1].recommendations_recieved[1:10])
print(ucb0.nodes[2].recommendations_recieved[1:10])
print(ucb0.nodes[3].recommendations_recieved[1:10])
print(ucb0.nodes[4].recommendations_recieved[1:10])

print(klucb0.nodes[0].recommendations_recieved[1:10])
print(klucb0.nodes[1].recommendations_recieved[1:10])
print(klucb0.nodes[2].recommendations_recieved[1:10])
print(klucb0.nodes[3].recommendations_recieved[1:10])
print(klucb0.nodes[4].recommendations_recieved[1:10])

# print(klucb1.nodes[4].when_recieved_arm(best_arm))
# print(klucb1.nodes[3].when_recieved_arm(best_arm))
# print(klucb1.nodes[2].when_recieved_arm(best_arm))
# print(klucb1.nodes[1].when_recieved_arm(best_arm))
# print(klucb1.nodes[0].when_recieved_arm(best_arm))

# print(ucb1.nodes[4].when_recieved_arm(best_arm))
# print(ucb1.nodes[3].when_recieved_arm(best_arm))
# print(ucb1.nodes[2].when_recieved_arm(best_arm))
# print(ucb1.nodes[1].when_recieved_arm(best_arm))
# print(ucb1.nodes[0].when_recieved_arm(best_arm))

# print(ucb1.adjust_comm_budget(range(1000))[1:100])


plt.plot(reg0, color='r', label='UCB')
plt.plot(klreg0, color='g', label='KL-UCB')
plt.xlabel("T")
plt.ylabel("Regret")
plt.legend()
plt.show()
