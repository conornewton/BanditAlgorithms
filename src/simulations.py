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
    comm_rounds = [20]
    # return [max(next(filter(lambda t: t >= i, comm_budget)), math.ceil((1 + i) ** (1 + eps))) for i in range(len(comm_budget))]
    while comm_rounds[-1] < length:
        comm_rounds.append(math.floor(len(comm_rounds) ** 3  + 20))
    return comm_rounds

def simulate(delta, high, low, alpha, t = 100000, k = 20, n = 5, iters = 10):
    start = timer()

    comm_rounds = adjust_comm_budget(t, 0.1)

    for i in range(iters):
        arms = param_arms(delta, high, low, k)
        unif = [[np.random.uniform() for _ in range(t)] for _ in range(k)]

        out_ucb = GosInE(n, arms, node_type = "UCB", gossip_matrix = "COMPLETE", alpha = alpha)
        out_ucb.play_unif(t, comm_rounds, unif),

        print(f"ucb{i}\t time taken: {timer() - start}")

        out_klucb = GosInE(n, arms, node_type = "KL-UCB", gossip_matrix = "COMPLETE", alpha = alpha)
        out_klucb.play_unif(t, comm_rounds, unif)

        print(f"klucb{i}\t time taken: {timer() - start}")

        pickle.dump(out_ucb, open(f"data/ucb_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "wb"))
        pickle.dump(out_klucb, open(f"data/klucb_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "wb"))

if __name__ == "__main__":
    task_num = int(sys.argv[1])

    width_min  = 0
    width_step = 0.025
    width_max  = 0.4

    # Distances between best arm and second best arm
    widths = [width_step * i for i in range(int((width_max - width_min) / width_step) + 1)]

    alphas = [1]

    param_array = list(itertools.product(widths, [0.9], [0.2], alphas))

    params = param_array[task_num % 17]
    if task_num < 17:
        simulate(params[0], params[1], params[2], params[3])
    elif task_num < 34:
        simulate(params[0], params[1], params[2], params[3], k = 30, n = 10)
    else:
        simulate(params[0], params[1], params[2], params[3], k = 50, n = 20)
