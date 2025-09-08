import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from helpers import *
from Sampler import Sampler
from tqdm import tqdm
import os
from helpers import calculate_uniform_average_action





# Cardinality of causal set that we are interested in
cardinality = 6

print("Calculating average BD action for cardinality", cardinality)
unique_bitstrings, unique_causal_bitstrings = get_unique_causal_bitstrings(cardinality)
average_BD_action = calculate_uniform_average_action(cardinality, causal_bitstrings = unique_causal_bitstrings, stdim=4, )
print("Average BD action for cardinality ", cardinality, " is ", average_BD_action)

num_qubits = [1,3,5,7]#1,]

num_chains = 2
num_samples = 100
sample_frequency = 1
T_therm = 0

# Store results for each chain
results = [[] for _ in range(len(num_qubits)+1)]#*(len(num_qubits)+1)
for i in range(num_chains):
    print(f"Starting {i}th round of chains")
    Csamp = Sampler(cardinality, method="classical", dimension=4, cargs={"link_move": True, "relation_move": True}, verbose = False)
    results[0].append(Csamp.sample_uniform(num_samples=num_samples, sample_frequency=sample_frequency, T_therm=T_therm, observables=["BD_action"]))

    #for j, nq in enumerate(num_qubits):
    for j in range(len(num_qubits)):
        
        if num_qubits[j] == 1:
            gamma_TC = 0.7
        elif num_qubits[j] == 3:
            gamma_TC = 0.9
        elif num_qubits[j] == 5:
            gamma_TC = 0.92
        elif num_qubits[j] == 7:
            gamma_TC = 0.93
        Qsamp = Sampler(cardinality, method="quantum", dimension=4, qargs={"gammas": [gamma_TC, 0, 1-gamma_TC], "t": 20, "num_qubits": num_qubits[j]}, verbose = True)
        results[j+1].append(Qsamp.sample_uniform(num_samples=num_samples, sample_frequency=sample_frequency, T_therm=T_therm, observables=["BD_action"]))

# Compute averages
def average_over_chains(results_list, key):
    arr = np.array([r[key] for r in results_list])
    mean = np.mean(arr, axis=0)
    sem = np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mean, sem
sample_index = results[0][0]["sample_index"]  # Assuming all have same indices



avg_bds = []
sem_bds = []
for i in range(len(num_qubits)+1):#len(num_qubits)+1):
    avg_bd, sem_bd = average_over_chains(results[i], "BD_action")
    avg_bds.append(avg_bd)
    sem_bds.append(sem_bd)
avg_bds = np.array(avg_bds)
sem_bds = np.array(sem_bds)


# Change working directory to 'plots'
os.makedirs("plots", exist_ok=True)
os.chdir("plots")
def plot_with_shaded_error_multi(x, ys, sems, labels, ylabel, title, filename):
    for y, sem, label in zip(ys, sems, labels):
        plt.plot(x, y, label=label)
        plt.fill_between(x, y - sem, y + sem, alpha=0.3)
    plt.xlabel("Sample index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# Plot BD action with shaded SEM (classical and quantum)

plt.axhline(average_BD_action, color='k', linestyle='--', label='Uniform Average BD Action')
plot_with_shaded_error_multi(
    sample_index,
    avg_bds,
    sem_bds,
    ["classical",]+ [f"{nq} qubits" for nq in num_qubits],
    "BD action", "Average BD action over 5 Markov chains",
    f"BD_action_avg_{cardinality}.png"
)

