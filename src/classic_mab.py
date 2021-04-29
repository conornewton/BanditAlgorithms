"""
    Plays the classic stochastic MAB on a given node type (UCB or KL-UCB)
"""

class ClassicMAB:
    """
        Class that plays an instance of a MAB problem with a single node and a set of arms
        given a node that makes decisions
    """
    def __init__(self, arms, node_type):
        self.arms = arms

    def play(self, t):
        """
            Plays t rounds of the MAB
        """
        pass

    def regret(self):
        """
            Returns an array containing the cumilative regret
        """
        pass
