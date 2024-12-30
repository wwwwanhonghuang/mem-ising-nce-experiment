import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Pool
from utils.utils import list_configutation
from pyclasses.ising_model import IsingModel, IsingModelIO
from pyclasses.ising_inferencer import IsingInferencer

def cost_function_single_ising_spin_configuration(partitioning_configuration, ising_spin_configuration, possibility_function, connection_function = None):
    def calculate_cross_partition_communication_cost(partitioning_configuration, ising_spin_configuration):
        cross_partition_communication_cost = 0
        n = len(ising_spin_configuration)
        for i in range(n):
            for j in range(n):
                if i != j and partitioning_configuration[i] != partitioning_configuration[j] and ising_spin_configuration[i] == 1 and (connection_function is None and (i + 1) % n == j) || (connection_function is not None):
                    cross_partition_communication_cost += 1 if (connection_function is None or connection_function(i, j)) else 0
        return cross_partition_communication_cost
    configuration_possibility = possibility_function(ising_spin_configuration)
    cost = configuration_possibility * calculate_cross_partition_communication_cost(partitioning_configuration, ising_spin_configuration)
    return cost

def cost_function_in_ising_spin_configurations(partitioning_configuration, ising_configuration_samples, possibility_function):
    cost = 0
    n = len(partitioning_configuration)
    for sample in ising_configuration_samples:
        if type(sample) is int:
            configuration =  list_configutation(sample, n)
        else:
            configuration = sample
        cost += cost_function_single_ising_spin_configuration(partitioning_configuration, configuration, possibility_function)
    return cost

def to_partitioning_configuration(permutation):
    avaliable_cores_per_chip = [4, 5, 5, 5, 4, 5, 5]
    n = len(permutation)
    if n > sum(avaliable_cores_per_chip):
        return None
    partitioning_configuration = [0] * n
    current_chip_id = 0
    for index, neuron_population_id in enumerate(permutation):
        while (avaliable_cores_per_chip[current_chip_id] == 0):
            current_chip_id += 1
        partitioning_configuration[neuron_population_id] = current_chip_id
        avaliable_cores_per_chip[current_chip_id] -= 1
    return partitioning_configuration


with open("4_energy_cost_evaluation/config.yaml", "r") as file:
    config = yaml.safe_load(file)

ising_model_file = config["evaluation"]["ising_model_file"]
Z1 = config["evaluation"]["Z1"]
Z2 = config["evaluation"]["Z2"]
log_2_n_samples = config["evaluation"]["log_2_n_samples"]
output_file = config["evaluation"]["output_file"]


ising_model = IsingModelIO.read_ising_model_from_file(ising_model_file)
inferencer = IsingInferencer(Z1 = Z1, Z2 = Z2)

energy_profiling_record_base_path = 
energy_record_files = os.listdir(energy_profiling_record_base_path)
energy_performances = [None] * len(energy_record_files)

ising_spin_configuration_samples = np.load("ising-spin-samples.npy")[-(1 << log_2_n_samples):, :]

energy_performances = {
}
def process_file(file):
    permutation = [int(element) for element in file.split("_")[1:]]
    partitioning_configuration = to_partitioning_configuration(permutation)
    
    cost = cost_function_in_ising_spin_configurations(partitioning_configuration, ising_spin_configuration_samples, 
        lambda ising_spin_configuration: inferencer.calculate_energy(ising_model, ising_spin_configuration, 2))
    return file, cost, partitioning_configuration

with Pool(processes=os.cpu_count()) as pool:
    # Use tqdm for progress tracking
    results = list(
        tqdm(pool.imap_unordered(process_file, energy_record_files), total=len(energy_record_files))
    )

for result in results:
    file, cost, partitioning_configuration = result
    if cost is not None:
        energy_performances[file] = {
            'file': file,
            'partitioning_configuration': partitioning_configuration,
            'cost_function_propose_cost': cost
        }
    else:
        print(f"Error processing file {file}: {partitioning_configuration}")

for performance in energy_performances.values():
    if performance:
        print(f"File: {performance['file']}, Cost: {performance['cost_function_propose_cost']}")

np.save(output_file, energy_performances, allow_pickle=True)
