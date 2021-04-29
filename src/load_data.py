import itertools
import pickle

from os import listdir

# width_min  = 0
# width_step = 0.025
# width_max  = 0.25

# # Distances between best arm and second best arm
# widths = [width_step * i for i in range(int((width_max - width_min) / width_step) + 1)]

# param_array = list(itertools.product(widths, [0.75, 0.7, 0.65, 0.55, 0.5], [0.5, 0.45, 0.4, 0.35, 0.3]))

# ucbs = []
# klucbs = []

# ucb_filespaths = [f"data/comm_ucb_100000_20_5_{param_array[i][0]}_{param_array[i][1]}_{param_array[i][2]}" for i in range(len(param_array))]

# klucb_filespaths = [f"data/comm_klucb_100000_20_5_{param_array[i][0]}_{param_array[i][1]}_{param_array[i][2]}" for i in range(len(param_array))]

# for i in range(10):
#     for j in range(len(ucb_filespaths)):
#         ucbs = pickle.load(open(ucb_filespaths[]))

# files = listdir("./data/")
# groups = []

# for i in range(len(files)):
#     for j in range(i, len(files)):
#         if files[i] == files[j]:
#             pass

def delta_values():
    width_min  = 0
    width_step = 0.025
    width_max  = 0.25

    widths = [width_step * i for i in range(int((width_max - width_min) / width_step) + 1)]

    return widths

def high_values():
    return [0.75, 0.7, 0.65, 0.6, 0.55, 0.5]

def low_values():
    return [0.5, 0.45, 0.4, 0.35, 0.3]

def load_data(delta, high, low, alpha, t = 100000, k = 20, n = 5, iters = 10, root_dir="./data"):
    ucb = []
    klucb = []

    for i in range(iters):
        ucb.append(pickle.load(open(f"{root_dir}/ucb_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "rb")))
        klucb.append(pickle.load(open(f"{root_dir}/klucb_{t}_{k}_{n}_{delta:.2f}_{high:.2f}_{low:.2f}_{alpha:.2f}_{i}.p", "rb")))

    return (ucb, klucb)
