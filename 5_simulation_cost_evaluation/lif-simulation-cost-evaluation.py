import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse
import pandas as pd
import re
from utils.utils import configuration_int32_encoding
import yaml

def read_from_folder(path):
    records = []
    for file_name in os.listdir(path):
        record = pd.read_csv(os.path.join(path, file_name), header=None)
        record["file_name"] = file_name
        records.append(record)
    return records

def to_population_spike_time(record):
    import numbers
    
    spike_times = set()
    for col_name in record:
        if(col_name == "file_name"):
            continue
        for spike_time in record[col_name]:
            if isinstance(spike_time, numbers.Number):
                spike_times.add(spike_time)
    return sorted(list(spike_times))
    
    

with open("5_simulation_cost_evaluation/config.yaml", "r") as file:
    config = yaml.safe_load(file)

record_path = config["cost-sampling"].get("record_path", "./data/simulator-lif-spikes/logs-100000")
n_neuron_pops = config["cost-sampling"].get("n_neuron_pops", 29)
T = config["cost-sampling"].get("T", 100000)
output = config["cost-sampling"].get("output", "./data/observation_ising_spin_configuration_simulator_lif.bin")
n_deployment_samples = config["cost-sampling"].get("n_deployment_samples", 5000)

print(f"Record Path: {record_path}")
print(f"Number of Neuron Populations: {n_neuron_pops}")
print(f"T: {T}")
print(f"Output Path: {output}")

spike_record_file_path = [os.path.join(log_base_path, file_name) for file_name in list(filter(lambda file_path: file_path[-6:] == "spikes", os.listdir(log_base_path)))]

spike_record = {}
for file_path in spike_record_file_path:
    spike_emission_time = list(pd.read_csv(file_path)["t\\id"])
    spike_record[int(re.match(r'.*neuron_([0-9]+).*', file_path).group(1))] = spike_emission_time


 
evaluated_costs = {

}


for deployment_configuration_id in range(n_deployment_samples):
    neuron_populations = list(range(n_neuron_pops))
    shuffle(neuron_populations)
    configuration = np.zeros(n_neuron_pops, dtype=int)
    pos = 0
    available_cores = max_cores_per_chip[:]
    
    for i in range(n_neuron_pops):
        while available_cores[pos] <= 0:
            pos += 1
        configuration[neuron_populations[i]] = pos
        available_cores[pos] -= 1
    evaluated_costs[tuple(configuration)] = 0
    
 
n_time_steps = T

pointers = [0] * n_neuron_pops

core_indexes = []
for index, count in enumerate(max_cores_per_chip):
    core_indexes.extend([index] * count)

# O(T * S * n^2) approx O(70000 * 2000 * 900)
for t in tqdm(range(n_time_steps)):
    for deployment_configuration in evaluated_costs.keys():
	    for index1, neuron_id in enumerate(deployment_configuration):
		if neuron_id not in spike_record:
		    continue
		while pointers[neuron_id] < len(spike_record[neuron_id]) and spike_record[neuron_id][pointers[neuron_id]] < t:
		    pointers[neuron_id] += 1
		    
		if pointers[neuron_id] < len(spike_record[neuron_id]) and spike_record[neuron_id][pointers[neuron_id]] == t:
		    for index2, other_neuron_id in enumerate(deployment_configuration):
		    	if other_neuron_id == neuron_id:
		    		continue
		    	if(core_indexes[index1] !=  core_indexed[index2] and connected(neuron_id, other_neuron_id)):
		    		evaluated_costs[deployment_configuration] += 1


