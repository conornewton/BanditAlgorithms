import random

class GossipMatrix:
    def __init__(self, comm_matrix):
        self.matrix = comm_matrix

    def sample(self, i):
        return random.choices(range(len(self.matrix)), self.matrix[i], k = 1)[0]


# TODO: maybe do this the better way
class CompleteGossipMatrix(GossipMatrix):
    def __init__(self, n):
        self.matrix = []
        for i in range(n):
            row_i = []
            prob = 1 / (n - 1)
            for j in range(n):
                if j == i:
                    row_i.append(0)
                else:
                    row_i.append(prob)
            self.matrix.append(row_i)

class StarGossipMatrix(GossipMatrix):
    def __init__(self, n):
        self.n = n

    def sample(self, i):
        if i == 0:
            # Node 0 is connected to all other nodes
            # probs = [0] + [1.0 / (self.n - 1)  for i in range(self.n - 1)]
            # return random.choices(range(len(self.n)), self.n[i], k = 1)[0]
            return random.randint(1, self.n - 1)

        else:
            return 0 #Nodes are only connected to node 0

class RingGossipMatrix(GossipMatrix):
    def __init__(self, n):
        self.n = n

    def sample(self, i):
        """ Returns one of the neighbouring nodes at random"""
        n1 = (i + 1) % self.n
        n2 = (i - 1) % self.n

        return random.choice([n1, n2])
