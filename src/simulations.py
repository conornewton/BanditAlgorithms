#!/usr/bin/env python

import pickle
import sys
import numpy as np
import math
import itertools

from timeit import default_timer as timer
from arms import param_arms
from gie import GosInE

def adjust_comm_budget(length, eps):
    comm_rounds = [0]
    # return [max(next(filter(lambda t: t >= i, comm_budget)), math.ceil((1 + i) ** (1 + eps))) for i in range(len(comm_budget))]
    while comm_rounds[-1] < length:
        comm_rounds.append(math.floor(len(comm_rounds) ** 3))
    return comm_rounds

def simulate(delta, high, low):
    start = timer()
    t = 100000
    k = 20 # Number of arms
    n = 5  # Number of nodes
    iters = 10 # Number of times to repeat the simulation

    comm_rounds = adjust_comm_budget(100000, 0.1)

    for i in range(iters):
        arms = param_arms(delta, high, low, k)
        unif = [[np.random.uniform() for _ in range(t)] for _ in range(k)]

        out_ucb = GosInE(n, arms, node_type = "UCB", eps = 0.1, alpha = 4)
        out_ucb.play_unif(t, comm_rounds, unif),

        print(f"ucb{i}\t time taken: {timer() - start}")

        out_klucb = GosInE(n, arms, node_type = "KL-UCB", eps = 0.1)
        out_klucb.play_unif(t, comm_rounds, unif)

        print(f"klucb{i}\t time taken: {timer() - start}")

        pickle.dump(out_ucb, open(f"data/comm_ucb_{t}_{k}_{n}_{delta}_{high}_{low}_{i}.p", "wb"))
        pickle.dump(out_klucb, open(f"data/comm_klucb_{t}_{k}_{n}_{delta}_{high}_{low}_{i}.p", "wb"))

if __name__ == "__main__":
    task_num = int(sys.argv[1])

    width_min  = 0
    width_step = 0.025
    width_max  = 0.25

    # Distances between best arm and second best arm
    widths = [width_step * i for i in range(int((width_max - width_min) / width_step) + 1)]

    # len 275
    param_array = list(itertools.product(widths, [0.75, 0.7, 0.65, 0.55, 0.5], [0.5, 0.45, 0.4, 0.35, 0.3]))

    params = param_array[task_num]
    simulate(params[0], params[1], params[2])
