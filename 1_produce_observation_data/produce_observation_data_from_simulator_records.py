import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse
import pandas as pd
import re
from utils.utils import configuration_int32_encoding


parser = argparse.ArgumentParser()
parser.add_argument("--record_path", type=str, default="./data/simulator-lif-spikes/logs-100000")
parser.add_argument("--n_neuron_pops", type=int, default=29)
parser.add_argument("--T", type=int, default=100000)
parser.add_argument("--output", type=str, default="./data/observation_ising_spin_configuration_simulator_lif.bin")


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

log_base_path = args.record_path
spike_record_file_path = [os.path.join(log_base_path, file_name) for file_name in list(filter(lambda file_path: file_path[-6:] == "spikes", os.listdir(log_base_path)))]

spike_record = {}
for file_path in spike_record_file_path:
    spike_emission_time = list(pd.read_csv(file_path)["t\id"])
    spike_record[int(re.match(r'.*neuron_([0-9]+).*', file_path).group(1))] = spike_emission_time


n_time_steps = T
n_neuron_pops = args.n_neuron_pops
observation_data = np.zeros((T, n_neuron_pops))
pointers = [0] * n_neuron_pops
for t in tqdm(range(n_time_steps)):
    for neuron_id in range(n_neuron_pops):
        if neuron_id not in spike_record:
            continue
        while pointers[neuron_id] < len(spike_record[neuron_id]) and spike_record[neuron_id][pointers[neuron_id]] < t:
            pointers[neuron_id] += 1
        if pointers[neuron_id] < len(spike_record[neuron_id]) and spike_record[neuron_id][pointers[neuron_id]] == t:
            observation_data[t, neuron_id] = 1


encoded_spin_configurations = np.array([configuration_int32_encoding(configuration) for configuration in observation_data], dtype=np.int32)
with open(args.output, "wb") as f:
    encoded_spin_configurations.tofile(f)

spike_sum = np.sum(observation_data, axis = 0) 
print(spike_sum.shape)
print(f'observation average of <sigma> is:')
for i in range(n_neuron_pops):
	 print(f"\t <sigma[{i}]>_emp = {spike_sum [i] / T}")
