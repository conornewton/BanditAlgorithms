"""
    Utility functions for analysing the data
"""

import numpy as np

import matplotlib.pyplot as plt

def average_results(results):
    """
        Given a list of cumilative regret lists, average them
    """
    avg_res = np.zeros(len(results[0]))
    for res in results:
        avg_res = np.add(avg_res, res)

    avg_res = np.true_divide(avg_res, len(results))
    return avg_res

def plot_regret(ucb_regret, kl_regret):
    """
        Plot ucb vs klucb regret
    """
    plt.plot(ucb_regret, 'r')
    plt.plot(kl_regret, 'g')
    plt.xlabel("T")
    plt.ylabel("Regret")
    plt.show()
