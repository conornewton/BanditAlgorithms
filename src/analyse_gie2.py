import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from gie import GosInE


def load_regrets(delta, high, low, n = 20, k = 50, iters = 100, alpha = 1, root_dir="./data/gie2"):
    regrets = []
    t = 100000

    for i in range(iters):
        gie  = pickle.load(open(f"{root_dir}/gie_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "rb"))
        gie2 = pickle.load(open(f"{root_dir}/gie2_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "rb"))
        
        gie_reg = gie.average_regret()
        gie2_reg = gie2.average_regret()
        
        for j in range(0, 100000, 10000):
            regrets.append([j, "GIE", gie_reg[j]])
            regrets.append([j, "GIE2", gie2_reg[j]])
            
    return regrets

def plot_alphas(delta, high, low, n, k):
    
    regrets = load_regrets(delta, high, low, n, k, iters = 1)

    df = pd.DataFrame(regrets, columns = ["Time", "GIE-Type", "Regret"])
    sns.lineplot(data = df, x = "Time", y = "Regret", hue = "GIE-Type", markers = True, err_style="bars", ci = 95)
    plt.show()

plot_alphas(0.1, 0.9, 0.2, 5, 20)