"""
    Utility functions for analysing the data
"""

import numpy as np


def average_regret(nodes, max_mean):
    """
        Given a list of nodes return their average_regret
    """
    avg_regret = np.zeros(len(nodes[0].rewards))

    for node in nodes:
        avg_regret = np.add(avg_regret, node.regret(max_mean))

    avg_regret = np.true_divide(avg_regret, len(nodes))
    return avg_regret

def average_results(results):
    """
        Given a list of cumilative regret lists, average them
    """
    avg_res = np.zeros(len(results[0]))
    for res in results:
        avg_res = np.add(avg_res, res)

    avg_res = np.true_divide(avg_res, len(results))
    return avg_res