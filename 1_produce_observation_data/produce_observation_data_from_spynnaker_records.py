import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse
from utils.utils import configuration_int32_encoding

parser = argparse.ArgumentParser()
parser.add_argument("--record_path", type=str, default="./data/synn-profiled-spikes/spike-chain\spikes_st1000000_weight_0.75")
parser.add_argument("--n_neuron_pops", type=int, default=29)
parser.add_argument("--T", type=int, default=1000000)
parser.add_argument("--output", type=str, default="./data/observation_ising_spin_configuration_spike_chain.bin")

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
    
args = parser.parse_args()
T = args.T
records = read_from_folder(args.record_path)
population_spike_times = [to_population_spike_time(population_spike_record) for population_spike_record in records]

n_neuron_populations = args.n_neuron_pops

n_neuron_populations = len(population_spike_times)
observation_configurations = np.zeros((T, n_neuron_populations))

pointers = np.zeros(n_neuron_populations, dtype=int)

for t in tqdm(range(T)):
    for neuron_id in range(n_neuron_populations):
        # Move pointer until we find the next spike time >= t
        while pointers[neuron_id] < len(population_spike_times[neuron_id]) and population_spike_times[neuron_id][pointers[neuron_id]] < t:
            pointers[neuron_id] += 1

        # If we find an exact match of t and the current spike time, update the configuration
        if pointers[neuron_id] < len(population_spike_times[neuron_id]) and t == population_spike_times[neuron_id][pointers[neuron_id]]:
            observation_configurations[t, neuron_id] = 1

encoded_configurations = np.array([configuration_int32_encoding(configuration) for configuration in observation_configurations], dtype=np.int32)

with open(args.output, "wb") as f:
    encoded_configurations.tofile(f)


spike_sum = np.sum(observation_configurations, axis = 0) 
print(spike_sum.shape)
print(f'observation average of <sigma> is:')
for i in range(args.n_neuron_pops):
	 print(f"\t <sigma[{i}]>_emp = {spike_sum [i] / T}")