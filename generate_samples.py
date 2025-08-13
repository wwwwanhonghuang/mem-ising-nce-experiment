from neurons.lif_neuron import LIFNeuronPopulation
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from network_samples.reservoir import get_random_reservoir, get_biased_reservoir, get_spike_chain

from utils import evaluate_partitioning

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import yaml
import argparse
import os
from current_generator.interval_current_generator import IntervalCurrentGenerator 


parser = argparse.ArgumentParser()

parser.add_argument("--configuration-file-path")

args = parser.parse_args()

configuration_file_path = args.configuration_file_path
print(f'Use configuration file: {configuration_file_path}')

# Load YAML config
with open(configuration_file_path, 'r') as file:
    config = yaml.safe_load(file)

print("Loaded configuration:")
print(config)

# Network parameters
num_pops = config['network']['n_pops']
T = config['network']['simulation_times']
neurons_per_pop = config['network']['neurons_per_pop']
dt = config['network']['dt']
project_name = config['project']['name']

os.makedirs(os.path.join("projects", project_name, "images"), exist_ok=True)
os.makedirs(os.path.join("projects", project_name, "model_save"), exist_ok=True)
os.makedirs(os.path.join("projects", project_name, "data"), exist_ok=True)

images_root = os.path.join("projects", project_name, "images")
model_save_root = os.path.join("projects", project_name, "model_save")
data_root = os.path.join("projects", project_name, "data")

if config['network']['type'] == 'spike_chain':
    populations, connection_matrices = get_spike_chain(num_pops=num_pops, neurons_per_pop = neurons_per_pop, 
            dt = dt, weight_mean=config['network']['params']['weight_mean'], weight_std=config['network']['params']['weight_std'])
    print(f"create spike_chain network, with weight_mean = {config['network']['params']['weight_mean']}, weight_std={config['network']['params']['weight_std']}")
elif config['network']['type'] == 'reservoir':
    populations, connection_matrices = get_random_reservoir(num_pops=num_pops, neurons_per_pop = neurons_per_pop, 
            dt = dt, weight_mean=config['network']['params']['weight_mean'], weight_std=config['network']['params']['weight_std'])
    print(f"create reservoir network, with weight_mean = {config['network']['params']['weight_mean']}, weight_std={config['network']['params']['weight_std']}")
elif config['network']['type'] == 'bias_reservoir':
    populations, connection_matrices = get_biased_reservoir(num_pops=num_pops, neurons_per_pop = neurons_per_pop, 
            dt = dt, weight_mean=config['network']['params']['weight_mean'], weight_std=config['network']['params']['weight_std'])
    print(f"create bias reservoir network, with weight_mean = {config['network']['params']['weight_mean']}, weight_std={config['network']['params']['weight_std']}")
else:
    raise NotImplementedError

weights_all_neurons = []
for p in populations:
    weights_all_neurons.append(p.W)
    print(p.W.sum())
np.save(os.path.join(data_root, "inner_pops_connection_matrices.npy"), weights_all_neurons, allow_pickle=True)



time = np.arange(0, T, dt)

if config['network']['currents']['type'] == 'interval':
    select_neuron_populations = config['network']['currents']['select_neuron_populations']
    apply_current_times_limits = config['network']['currents']['apply_current_times_limits']
    current_generator = IntervalCurrentGenerator(num_pops=num_pops, n_times=T, neurons_per_pop=neurons_per_pop, select_neurons=select_neuron_populations, limit_current_times=apply_current_times_limits)
    I_ext = current_generator.generate_currents()

# Prepare recording arrays
V_trace = np.zeros((len(time), num_pops, neurons_per_pop))
spikes_record = np.zeros((len(time), num_pops, neurons_per_pop), dtype=bool)

# Initialize spike state for previous step (per population)
S_prev = np.zeros((num_pops, neurons_per_pop))

for t_i, t in tqdm(enumerate(time)):
    # Compute inputs to each population from all populations
    inputs = np.zeros((num_pops, neurons_per_pop))
    for target_pop in range(num_pops):
        # Sum over all source populations
        for source_pop in range(num_pops):
            inputs[target_pop] += connection_matrices[target_pop, source_pop] @ S_prev[source_pop]
    
    # Add external input
    total_input = inputs + I_ext[t_i]
    
    # Step populations independently
    for p in range(num_pops):
        spikes = populations[p].step(total_input[p], t)
        V_trace[t_i, p] = populations[p].V
        spikes_record[t_i, p] = spikes
        S_prev[p] = spikes
    
print("Simulation finished.")

spike_times_all = []
neuron_indices_all = []

for pop_idx in range(num_pops):
    for neuron_idx in range(neurons_per_pop):
        # Get spike times for this neuron
        stimes = populations[pop_idx].spike_times[neuron_idx]
        # Global neuron ID (pop * neurons_per_pop + neuron)
        global_id = pop_idx * neurons_per_pop + neuron_idx
        spike_times_all.extend(stimes)
        neuron_indices_all.extend([global_id] * len(stimes))

spike_times_all = np.array(spike_times_all)
neuron_indices_all = np.array(neuron_indices_all)

# Plot raster
plt.figure(figsize=(15, 10))
plt.scatter(spike_times_all, neuron_indices_all, s=1, color='black')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index (populations stacked)')
plt.title('Raster plot of all neurons in {n_pops} populations (7424 neurons)')
plt.ylim([-1, num_pops*neurons_per_pop])

plt.savefig(os.path.join(images_root, config['network']['image_store']['raster_image']))
np.savez(os.path.join(data_root, config['network']['spike_data_store']['path']), 
         neuron_indices_all=neuron_indices_all, spike_times_all=spike_times_all)
np.save(os.path.join(data_root, "inter_pops_connection_matrices.npy"), connection_matrices)



num_trials = 0 # number of random assignments to test
if num_trials == 0:
    exit()
num_pops = len(populations)
num_chips = 3
cores_per_chip = 4
num_cores = num_chips * cores_per_chip

all_metrics = []  # list to store all metrics dict per trial
J = np.zeros((num_pops, num_pops))
# Make symmetric if functional coupling is symmetric
J = (J + J.T) / 2

for trial in range(num_trials):
    chip_ids = np.random.randint(0, num_chips, size=num_pops)
    core_ids = np.random.randint(0, cores_per_chip, size=num_pops)
    partitioning = np.column_stack((chip_ids, core_ids))
    random_assignments = chip_ids * cores_per_chip + core_ids

    metrics = evaluate_partitioning(
        populations,
        random_assignments,
        J,
        neuron_update_cost=1.0,
        spiking_cost=5.0,
        total_frequency_budget=80.0,
        f_min=2.0,
        f_max=12.0,
        comm_bandwidth_matrix=None,
        power_per_freq_unit=1.5,
    )

    all_metrics.append(metrics)

    print(f"Trial {trial + 1} Metrics:")
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"  {k}: {v:.4f}")
        else:
            if hasattr(v, 'shape'):
                print(f"  {k}: array shape {v.shape}, first 5: {v[:5]}")
            else:
                print(f"  {k}: {v}")
    print("\n")


num_trials = 1000  # number of random assignments to test

# After you run the trials and have all_metrics collected as before...

# First, identify all metric keys
metric_keys = all_metrics[0].keys()

# Prepare a dict to hold reduced stats per metric across trials
reduced_metrics = {k: [] for k in metric_keys}

for m in all_metrics:
    for k, v in m.items():
        # If scalar, just append
        if isinstance(v, (int, float)):
            reduced_metrics[k].append(v)
        else:
            # If numpy array or list, convert to numpy array and take mean
            arr = np.array(v)
            reduced_metrics[k].append(arr.mean())

num_metrics = len(metric_keys)
cols = 3
rows = (num_metrics + cols - 1) // cols

plt.figure(figsize=(5 * cols, 4 * rows))

for i, k in enumerate(metric_keys, 1):
    data = np.array(reduced_metrics[k])
    mean_val = data.mean()
    std_val = data.std()

    plt.subplot(rows, cols, i)
    plt.hist(data, color='tab:blue', alpha=0.7, density=True)
    
    if std_val > 1e-8:
        try:
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min() - std_val, data.max() + std_val, 200)
            plt.plot(x_range, kde(x_range), color='darkorange', lw=2, label='KDE')
        except np.linalg.LinAlgError:
            # In case of unexpected numerical issues, skip KDE
            pass
    
    plt.axvline(mean_val, color='red', linestyle='-', label='Mean')
    plt.axvline(mean_val - std_val, color='red', linestyle='--', label='Mean ± 1 Std')
    plt.axvline(mean_val + std_val, color='red', linestyle='--')
    plt.title(k)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(images_root, "metrics.png"))
