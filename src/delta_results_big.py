import pickle
import numpy as np

from gie import GosInE
from simulation_util import *

from load_data import *


def plot_deltas(n, k):
    delta_min = 0
    delta_step = 0.025
    delta_max = 0.4
    deltas = [delta_step * i for i in range(int((delta_max - delta_min) / delta_step) + 1)]
    
    deltas.pop(0)
    
    ucb_regrets = []
    klucb_regrets = []
    
    for delta in deltas:
        # Load data
        (ucb_data, klucb_data) = load_data(delta=delta, high=0.9, low=0.2, alpha = 1, n = n, k = k, iters = 20, root_dir="./data/delta_results")

        # Average Results
        ucb_regrets.append(average_results([ucb.average_regret() for ucb in ucb_data])[-1])
        klucb_regrets.append(average_results([klucb.average_regret() for klucb in klucb_data])[-1])

    # Plot Data
    plt.plot(deltas, ucb_regrets, "ro", color="r", label="UCB")
    plt.plot(deltas, klucb_regrets, "ro", color="g", label="KL-UCB")
    plt.xlabel(r"$\Delta_2$")
    plt.ylabel("Regret")
    plt.legend()
    plt.show()

plot_deltas(20, 50)