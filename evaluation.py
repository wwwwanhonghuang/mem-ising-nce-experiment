import argparse
import yaml
import os
import re
import pickle
from ising_models.ising_trainer import TrainingContext, PairwiseIsingModel
from neurons.lif_neuron import LIFNeuronPopulation
from current_generator.interval_current_generator import IntervalCurrentGenerator 

from evaluation.simulation import *
import numpy as np
from tqdm import tqdm

import pickle

from utils import evaluate_partitioning_core_only, evaluate_partitioning_core_chip

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
partition_save_root = os.path.join("projects", project_name, "partition_schemes")
evaluation_results_save_root = os.path.join("projects", project_name, "evaluation_results")

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
    method: np.load(os.path.join(partition_save_root, f'partition_scheme_core_only_{method}.npy'), allow_pickle=True)[0]
    for method in partition_methods
}

def evaluate_core_only(partitioning_scheme, spike_times_all, neuron_indices_all):
    assignments = np.zeros((n_sites))
    for partiton_id, configuration in enumerate(partitioning_scheme):
        for pop_id in configuration:
            assignments[pop_id] = partiton_id
    evaluate_partitioning_core_only(populations=populations, assignments=assignments)
    

def random_partitioning_and_mapping_core_only(num_neurons: int, K: int):
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


def random_partitioning_and_mapping_core_and_chip(
    num_neurons_core: int,
    num_neurons_chip: int,
    K_core: int,
    K_chip: int
):
    """
    Randomly partition neurons into 'core' and 'chip' groups with separate limits.

    Args:
        num_neurons_core : int
            Number of neurons to assign to core partitions.
        num_neurons_chip : int
            Number of neurons to assign to chip partitions.
        K_core : int
            Maximum neurons per core node.
        K_chip : int
            Maximum neurons per chip node.

    Returns:
        Tuple[List[List[int]], List[List[int]]]:
            (core_parts, chip_parts)
            Each is a list of groups (lists of neuron indices).
    """
    # Shuffle neuron indices
    all_core_indices = np.random.permutation(num_neurons_core)
    all_chip_indices = np.random.permutation(num_neurons_chip)
    
    # Split into chunks for core
    core_parts = [list(all_core_indices[i:i + K_core]) 
                  for i in range(0, num_neurons_core, K_core)]
    
    # Split into chunks for chip
    chip_parts = [list(all_chip_indices[i:i + K_chip]) 
                  for i in range(0, num_neurons_chip, K_chip)]
    
    return core_parts, chip_parts


records = {
    method: {} for method in partition_methods
} | {'random': {}, 'random_core_chip': {}}


T = n_simulation_times
def run_simulations(n_simulation_trails=1000, record_key="", partitioning_strategy='random', frequency_strategy='fixed', max_pop_per_core=2, n_cores_per_chip=4, n_chips=3):
    records[record_key] = {}
    for trail_id in tqdm(range(n_simulation_trails), desc='simulating'):
        
        if config['network']['currents']['type'] == 'interval':
            select_neuron_populations = config['network']['currents']['select_neuron_populations']
            apply_current_times_limits = config['network']['currents']['apply_current_times_limits']
            current_generator = IntervalCurrentGenerator(num_pops=num_pops, n_times=T, neurons_per_pop=neurons_per_pop, select_neurons=select_neuron_populations, limit_current_times=apply_current_times_limits)
            I_ext = current_generator.generate_currents()
        
        network = Network(num_pops=num_pops, neurons_per_pop=neurons_per_pop, connection_matrices=connection_matrices, populations=populations)
        if frequency_strategy == 'fixed':
            hardware_configuration = HardwareConfiguration(n_cores=n_cores_per_chip, n_chips=n_chips, frequency_configuration= [[1000] * n_cores_per_chip] * n_chips)
        hardware = VirtualNeuromorphicHardware(hardware_configuration)
        
        if partitioning_strategy == 'random':
            deployment_configuration = random_partitioning_and_mapping_core_only(num_neurons=num_pops, K=max_pop_per_core)
            
        print(f"Deployment configuration == {deployment_configuration}")
        spike_times_all, neuron_indices_all = hardware.do_simulation(dt=dt, n_simulation_times=n_simulation_times, deployment_configuration=deployment_configuration, network=network, I_ext=I_ext)
        
        # for method in partition_methods:
        #     results = evaluate_core_only(partitioning_scheme=partition_results[method], spike_times_all=spike_times_all, neuron_indices_all=neuron_indices_all)
        #     records[method][trail_id] =  results
        
        hardware.performance_monitor.report_performance()
        records[record_key][trail_id] = dict(hardware.performance_monitor.cost)
        
        # records['random'][trail_id] = []
        # for _ in range(random_partitioning_count_each_trail):
        #     results = evaluate_core_only(partitioning_scheme=random_partitioning_and_mapping_core_only(), spike_times_all=spike_times_all, neuron_indices_all=neuron_indices_all)
        #     records['random'][trail_id].append(results)
            
            
        # for method in partition_methods:
        #     results = evaluate_partitioning_core_chip(partitioning_scheme=partition_results[method], spike_times_all=spike_times_all, neuron_indices_all=neuron_indices_all)
        #     records[method][trail_id] =  results
        
        # records['random_core_chip'][trail_id] = []
        # for _ in range(random_partitioning_count_each_trail):
        #     results = evaluate_partitioning_core_chip(partitioning_scheme=random_partitioning_and_mapping_core_only(), spike_times_all=spike_times_all, neuron_indices_all=neuron_indices_all)
        #     records['random_core_chip'][trail_id].append(results)


def evaluation_entry(type, n_trails):
    if type == "fixed_freq_maximum_random_partitioning":
        run_simulations(n_trails, record_key = 'fixed_freq_maximum_random_partitioning', frequency_strategy='fixed', partitioning_strategy='random')
    elif type == "variable_freq_avg_1std_random_partitioning":
        raise NotImplementedError
    elif type == "variable_freq_avg_2std_random_partitioning":
        raise NotImplementedError
    elif type == "variable_freq_avg_m1std_random_partitioning":
        raise NotImplementedError
    elif type == "variable_freq_avg_m2std_random_partitioning":
        raise NotImplementedError
    elif type == "variable_freq_ising_partitioning":
        raise NotImplementedError

print(f'evaluation finished.')

# with open(os.path.join(evaluation_results_save_root, "evaluation_record.pkl"), 'wb') as f:
#     pickle.dump(records, f)
