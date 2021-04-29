#!/usr/bin/env python

import pickle
import sys
import numpy as np
import math
import itertools

from timeit import default_timer as timer
from arms import param_arms
from gie import GosInE
from gie2 import GosInE2

def adjust_comm_budget(length, eps):
    comm_rounds = [0]
    while comm_rounds[-1] < length:
        comm_rounds.append(math.floor(len(comm_rounds) ** 3))
    return comm_rounds

def simulate_comm(delta, high, low, alpha, i, t = 100000, k = 50, n = 20, file_path = "./data/gie2conn"):
    comm_rounds = adjust_comm_budget(t, 0.1)

    arms = param_arms(delta, high, low, k)
    unif = [[np.random.uniform() for _ in range(t)] for _ in range(k)]

    complete= GosInE(n, arms, node_type = "KL-UCB", gossip_matrix = "COMPLETE", alpha = alpha)
    complete.play_unif(t, comm_rounds, unif)

    star = GosInE(n, arms, node_type = "KL-UCB", gossip_matrix = "STAR", alpha = alpha)
    star.play_unif(t, comm_rounds, unif)

    ring = GosInE(n, arms, node_type = "KL-UCB", gossip_matrix = "RING", alpha = alpha)
    ring.play_unif(t, comm_rounds, unif)

    pickle.dump(complete, open(f"data/complete_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "wb"))
    pickle.dump(star, open(f"data/star_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "wb"))
    pickle.dump(ring, open(f"data/ring_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "wb"))

def simulate2(delta, high, low, alpha, i, t = 100000, k = 20, n = 5, file_path = "./data/gie2"):
    comm_rounds = adjust_comm_budget(t, 0.1)
    arms = param_arms(delta, high, low, k)
    unif = [[np.random.uniform() for _ in range(t)] for _ in range(k)]

    gie = GosInE(n, arms, node_type = "KL-UCB", gossip_matrix = "COMPLETE", alpha = alpha)
    gie.play_unif(t, comm_rounds, unif)

    gie2 = GosInE2(n, arms, node_type = "KL-UCB", gossip_matrix = "COMPLETE", alpha = alpha)
    gie2.play_unif(t, comm_rounds, unif)

    pickle.dump(gie, open(f"{file_path}/gie_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "wb"))
    pickle.dump(gie2, open(f"{file_path}/gie2_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "wb"))

def simulate(delta, high, low, alpha, i, t = 100000, k = 20, n = 5):
    comm_rounds = adjust_comm_budget(t, 0.1)

    arms = param_arms(delta, high, low, k)
    unif = [[np.random.uniform() for _ in range(t)] for _ in range(k)]

    # out_ucb = GosInE(n, arms, node_type = "UCB", gossip_matrix = "COMPLETE", alpha = alpha)
    # # out_ucb.play_unif(t, comm_rounds, unif),

    # out_klucb = GosInE(n, arms, node_type = "KL-UCB", gossip_matrix = "COMPLETE", alpha = alpha)
    # out_klucb.play_unif(t, comm_rounds, unif),

    # out_klucb_index = GosInE(n, arms, node_type = "KL-UCB", gossip_matrix = "COMPLETE", alpha = alpha)
    # out_klucb_index.play_unif_index(t, comm_rounds, unif),

    # pickle.dump(out_klucb_index, open(f"data/klucb_index_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "wb"))
    # pickle.dump(out_klucb, open(f"data/klucb_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "wb"))
    # print(f"ucb{i}\t time taken: {timer() - start}")

    complete= GosInE(n, arms, node_type = "KL-UCB", gossip_matrix = "COMPLETE", alpha = alpha)
    complete.play_unif(t, comm_rounds, unif)

    star = GosInE(n, arms, node_type = "KL-UCB", gossip_matrix = "STAR", alpha = alpha)
    star.play_unif(t, comm_rounds, unif)

    ring = GosInE(n, arms, node_type = "KL-UCB", gossip_matrix = "RING", alpha = alpha)
    ring.play_unif(t, comm_rounds, unif)

    pickle.dump(complete, open(f"data/complete_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "wb"))
    pickle.dump(star, open(f"data/star_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "wb"))
    pickle.dump(ring, open(f"data/ring_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "wb"))

if __name__ == "__main__":
    task_num = int(sys.argv[1])

    # width_min  = 0
    # width_step = 0.025
    # width_max  = 0.4

    # Distances between best arm and second best arm
    # widths = [width_step * i for i in range(int((width_max - width_min) / width_step) + 1)]

    alphas = [1]

    param_array = list(itertools.product([0.1], [0.9], [0.2], alphas))
    params = param_array[0]

    simulate_comm(0.1, 0.9, 0.2, 1, task_num % 100, n = 20, k = 50)