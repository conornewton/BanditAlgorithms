import random

class GossipMatrix:
    def __init__(self, comm_matrix):
        self.matrix = comm_matrix

    def sample(self, i):
        return random.choices(range(len(self.matrix)), self.matrix[i], k = 1)[0]

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

class RingGossipMatrix(GossipMatrix):
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
