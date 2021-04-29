class BatchSimulation:
    def __init__(self, t, k, n, iters):
        self.t = t
        self.k = k
        self.n = n
        self.iters = iters


    def simulate(self, delta, high, low, alpha):
        for i in range(iters):
            arms = param_arms(delta, high, low, k)
            unif = [[np.random.uniform() for _ in range(t)] for _ in range(k)]

            # out_ucb = GosInE(n, arms, node_type = "UCB", gossip_matrix = "RING", alpha = 4)
            # out_ucb.play_unif(t, comm_rounds, unif),

            # print(f"ucb{i}\t time taken: {timer() - start}")

            out_klucb = GosInE(n, arms, node_type = "KL-UCB", gossip_matrix = "COMPLETE", alpha = 1)
            out_klucb.play_unif(t, comm_rounds, unif)

            print(f"klucb{i}\t time taken: {timer() - start}")

            out_klucbindex = GosInE(n, arms, node_type = "KL-UCB", gossip_matrix = "COMPLETE", alpha = 1)
            out_klucbindex.play_unif_index(t, comm_rounds, unif)

            print(f"klucbindex{i}\t time taken: {timer() - start}")

            pickle.dump(out_klucbindex, open(f"data/klucb_index_{t}_{k}_{n}_{delta}_{high}_{low}_{i}.p", "wb"))
            pickle.dump(out_klucb, open(f"data/klucb_{t}_{k}_{n}_{delta}_{high}_{low}_{i}.p", "wb"))
