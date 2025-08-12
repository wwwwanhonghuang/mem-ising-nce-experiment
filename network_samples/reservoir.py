


from neurons.lif_neuron import LIFNeuronPopulation
import numpy as np

def get_random_reservoir(num_pops=29, neurons_per_pop = 256, dt = 1.0):
    # Create populations
    populations = [LIFNeuronPopulation(neurons_per_pop, dt=dt, noise_std=0.7) for _ in range(num_pops)]
    
    # Create connection matrices: from population j -> population i, shape (num_pops, num_pops, neurons_per_pop, neurons_per_pop)
    # Random weights, e.g. normal with small std
    connection_matrices = np.random.normal(0, 0.05, (num_pops, num_pops, neurons_per_pop, neurons_per_pop))
    return populations, connection_matrices

def get_biased_reservoir(num_pops=29, neurons_per_pop=256, dt=1.0):
    populations = [LIFNeuronPopulation(neurons_per_pop, dt=dt, noise_std=0.7) for _ in range(num_pops)]

    # Initialize connection matrices with small random noise
    connection_matrices = np.random.normal(0, 0.01, (num_pops, num_pops, neurons_per_pop, neurons_per_pop))

    # Define modules, e.g. 3 modules roughly equal size
    module_sizes = []
    unit = int(np.ceil(num_pops / 3.0))
    total = 0
    while total < num_pops:
        if total + unit <= num_pops:
            module_sizes += [unit]
            total += unit
        else:
            module_sizes += [num_pops - unit]
            total += num_pops - unit
    
    module_indices = []
    start = 0
    for size in module_sizes:
        module_indices.append(list(range(start, start+size)))
        start += size

    # Increase connection strength within modules
    for m in module_indices:
        for i in m:
            for j in m:
                # Add stronger weights for intra-module connections
                connection_matrices[i, j] += np.random.normal(0.05, 0.01, (neurons_per_pop, neurons_per_pop))

    return populations, connection_matrices

def get_spike_chain(num_pops = 29, neurons_per_pop=256, dt=1.0, chain_weight_mean=3.5, chain_weight_std=1.2):
    populations = [LIFNeuronPopulation(neurons_per_pop, dt=dt, noise_std=0.7) for _ in range(num_pops)]

    # Base weak random connections
    connection_matrices = np.zeros((num_pops, num_pops, neurons_per_pop, neurons_per_pop))

   

    for i in range(num_pops):
        j = (i + 1) % num_pops  # next population in the chain, wraps around
        # Make strong connections from pop i -> pop j
        connection_matrices[j, i] = np.random.normal(chain_weight_mean, chain_weight_std, (neurons_per_pop, neurons_per_pop))

    return populations, connection_matrices