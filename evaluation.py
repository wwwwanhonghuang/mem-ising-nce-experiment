import argparse
import yaml
import os
import re
import pickle
from ising_models.ising_trainer import TrainingContext, PairwiseIsingModel
from neurons.lif_neuron import LIFNeuronPopulation
from current_generator.interval_current_generator import IntervalCurrentGenerator 

import numpy as np
import tqdm

import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--configuration-file-path", type=str)

args = parser.parse_args()


# Logics for loading  the best ising model's parameters J

configuration_file_path = args.configuration_file_path
print(f'Use configuration file: {configuration_file_path}')

# Load YAML config
with open(configuration_file_path, 'r') as file:
    config = yaml.safe_load(file)

print("loaded configurations:")
print(config)

n_sites = config['network']['n_pops']
project_name = config['project']['name']
num_pops = n_sites

images_root = os.path.join("projects", project_name, "images")
model_save_root = os.path.join("projects", project_name, "model_save")
data_root = os.path.join("projects", project_name, "data")
image_save_path_root = images_root

n_epoches = config['ising_training']['n_epoches']
neurons_per_pop = config['network']['neurons_per_pop']
record_file_path = os.path.join(data_root, config['network']['spike_data_store']['path'])
model_save_path_root = model_save_root

dt = config['network']['dt']


weight_mean=config['network']['params']['weight_mean']
weight_std=config['network']['params']['weight_std']
partition_save_root = os.path.join("project", project_name, "partition_schemes")
evaluation_results_save_root = os.path.join("project", project_name, "evaluation_results")

## Evaluation 1 - original network
save_weights = np.load(os.path.join(data_root, 'inner_pops_connection_matrices.npy'), allow_pickle=True)

def load_original_network():
    # Create populations
    populations = [LIFNeuronPopulation(neurons_per_pop, dt=dt, noise_std=0.7, W=save_weights[i], weight_mean=weight_mean, weight_std=weight_std) for i in range(n_sites)]
    connection_matrices = np.load(os.path.join(data_root, 'inter_pops_connection_matrices.npy'), allow_pickle=True)
    
    return populations, connection_matrices

populations, connection_matrices = load_original_network()



n_simulation_trails = 1000
n_simulation_times = 2000
random_partitioning_count_each_trail = 1000

partition_methods = ['greedy', 'agglomerative', 'spectral', 'simple_kmedoids', 'louvain', 'kernighan_lin']

partition_results = {
    method: np.load(os.path.join(partition_save_root, f'partition_scheme_core_only_{method}.npz'), allow_pickle=True)['arr_0']
    for method in partition_methods
}


def evaluate(partitioning_scheme, spike_times_all, neuron_indices_all):
    pass

def random_partitioning_and_mapping(num_neurons: int, K: int):
    """
    Randomly partition `num_neurons` neurons into groups of size <= K.

    Args:
        num_neurons : int
            Total number of neurons to partition.
        K : int
            Maximum number of neurons per group.

    Returns:
        List[List[int]] : List of groups (each group is a list of neuron indices)
    """
    # Shuffle neuron indices
    neuron_indices = np.random.permutation(num_neurons)
    
    # Split into chunks of size <= K
    parts = [list(neuron_indices[i:i + K]) for i in range(0, num_neurons, K)]
    
    return parts


records = {
    method: {} for method in partition_methods
} | {'random': {}}

def run_simulations(n_simulation_trails=1000):
    for trail_id in tqdm(range(n_simulation_trails), desc='simulating'):
        if config['network']['currents']['type'] == 'interval':
            select_neuron_populations = config['network']['currents']['select_neuron_populations']
            apply_current_times_limits = config['network']['currents']['apply_current_times_limits']
            current_generator = IntervalCurrentGenerator(num_pops=num_pops, n_times=T, neurons_per_pop=neurons_per_pop, select_neurons=select_neuron_populations, limit_current_times=apply_current_times_limits)
            I_ext = current_generator.generate_currents()
        time = np.arange(0, n_simulation_times, dt)

        V_trace = np.zeros((len(time), num_pops, neurons_per_pop))
        spikes_record = np.zeros((len(time), num_pops, neurons_per_pop), dtype=bool)

        # Initialize spike state for previous step (per population)
        S_prev = np.zeros((num_pops, neurons_per_pop))

        for t_i, t in tqdm(enumerate(n_simulation_times)):
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
        
        
        
        for method in partition_methods:
            results = evaluate(partitioning_scheme=partition_results[method], spike_times_all=spike_times_all, neuron_indices_all=neuron_indices_all)
            records[method][trail_id] =  results
        
        records['random'][trail_id] = []
        for _ in range(random_partitioning_count_each_trail):
            results = evaluate(partitioning_scheme=random_partitioning_and_mapping(), spike_times_all=spike_times_all, neuron_indices_all=neuron_indices_all)
            records['random'][trail_id].append(results)

print(f'evaluation finished.')

with open(os.path.join(evaluation_results_save_root, "evaluation_record.pkl"), 'wb') as f:
    pickle.dump(records, f)

## Evaluation 2

