import numpy as np

def evaluate_partitioning(
    populations,
    assignments,
    J,
    neuron_update_cost=1.0,
    spiking_cost=5.0,
    total_frequency_budget=100.0,
    f_min=1.0,
    f_max=10.0,
    comm_bandwidth_matrix=None,  # shape (num_cores, num_cores), optional
    power_per_freq_unit=1.0,     # simple linear power per frequency unit per core
):
    num_pops = len(populations)
    cores = np.unique(assignments)
    num_cores = len(cores)
    
    # 1. Communication Cost
    comm_cost = 0.0
    for i in range(num_pops):
        for j in range(num_pops):
            if assignments[i] != assignments[j]:
                comm_cost += J[i, j]
    
    # 2. Functional Coherence and 4. Load Balance Preparations
    intra_coupling = np.zeros(num_cores)
    population_counts = np.zeros(num_cores)
    workload_per_core = np.zeros(num_cores)
    spike_count_per_core = np.zeros(num_cores)
    
    for idx, core in enumerate(cores):
        # Populations in this core
        pops_in_core = np.where(assignments == core)[0]
        population_counts[idx] = len(pops_in_core)
        
        # Sum intra-core functional coupling
        core_J = J[np.ix_(pops_in_core, pops_in_core)]
        if len(pops_in_core) > 0:
            intra_coupling[idx] = np.sum(core_J) / (len(pops_in_core)**2)
        else:
            intra_coupling[idx] = 0.0
        
        # Calculate workload: neuron updates + spikes
        n_neurons = sum(populations[p].N for p in pops_in_core)
        n_spikes = sum(len(neuron_spikes) for p in pops_in_core for neuron_spikes in populations[p].spike_times)

        workload_per_core[idx] = n_neurons * neuron_update_cost + n_spikes * spiking_cost
        
        spike_count_per_core[idx] = n_spikes
    
    # 3. Load Balance (Coefficient of Variation)
    workload_mean = np.mean(workload_per_core)
    workload_std = np.std(workload_per_core)
    load_balance = workload_std / workload_mean if workload_mean > 0 else 0.0
    
    # 5. Cross-Partition Interference (simplified)
    cross_partition_interference = 0.0
    if comm_bandwidth_matrix is not None:
        for i in range(num_cores):
            for j in range(num_cores):
                if i != j:
                    # Sum J between populations in core i and core j
                    pops_i = np.where(assignments == cores[i])[0]
                    pops_j = np.where(assignments == cores[j])[0]
                    inter_J = np.sum(J[np.ix_(pops_i, pops_j)])
                    bandwidth = comm_bandwidth_matrix[i, j]
                    if bandwidth > 0:
                        cross_partition_interference += inter_J / bandwidth
    
    # 6. Power/Energy Estimate (simple linear)
    # Power ~ frequency * power_per_freq_unit + communication power proportional to comm cost
    # For now, estimate frequency allocation below
    
    # Frequency allocation optimization (simple proportional allocation)
    # Allocate frequencies proportional to workload, clipped by min/max and normalized to total budget
    freq_alloc = workload_per_core.copy()
    total_workload = np.sum(workload_per_core)
    if total_workload > 0:
        freq_alloc = freq_alloc / total_workload * total_frequency_budget
    freq_alloc = np.clip(freq_alloc, f_min, f_max)
    
    # Compute time cost per core
    time_cost_per_core = workload_per_core / freq_alloc
    
    # Total simulation time is max core time
    total_sim_time = np.max(time_cost_per_core)
    
    # Power estimate
    power_compute = np.sum(freq_alloc) * power_per_freq_unit
    power_comm = comm_cost  # simplified: proportional to communication cost
    total_power = power_compute + power_comm
    
    # Resource Utilization Efficiency (workload / allocated frequency sum)
    resource_util_eff = total_workload / np.sum(freq_alloc) if np.sum(freq_alloc) > 0 else 0.0
    
    # Package all metrics
    metrics = {
        'communication_cost': comm_cost,
        'intra_partition_coherence': np.mean(intra_coupling),
        'load_balance_cv': load_balance,
        'cross_partition_interference': cross_partition_interference,
        'total_simulation_time': total_sim_time,
        'power_estimate': total_power,
        'resource_utilization_efficiency': resource_util_eff,
        'frequency_allocation': freq_alloc,
        'time_cost_per_core': time_cost_per_core,
        'workload_per_core': workload_per_core,
        'spike_count_per_core': spike_count_per_core,
    }
    
    return metrics

def evaluate_partitioning_core_chip(populations,
    assignments,
    J,
    neuron_update_cost=1.0,
    spiking_cost=5.0,
    total_frequency_budget=100.0,
    f_min=1.0,
    f_max=10.0,
    comm_bandwidth_matrix=None,  # shape (num_cores, num_cores), optional
    power_per_freq_unit=1.0,     # simple linear power per frequency unit per core
    ):
    pass

def evaluate_partitioning_core_only(
    populations,
    assignments,
    J,
    neuron_update_cost=1.0,
    spiking_cost=5.0,
    total_frequency_budget=100.0,
    f_min=1.0,
    f_max=10.0,
    comm_bandwidth_matrix=None,  # shape (num_cores, num_cores), optional
    power_per_freq_unit=1.0,     # simple linear power per frequency unit per core
):
    num_pops = len(populations)
    cores = np.unique(assignments)
    num_cores = len(cores)
    
    # 1. Communication Cost
    comm_cost = 0.0
    for i in range(num_pops):
        for j in range(num_pops):
            if assignments[i] != assignments[j]:
                comm_cost += J[i, j]
    
    # 2. Functional Coherence and 4. Load Balance Preparations
    intra_coupling = np.zeros(num_cores)
    population_counts = np.zeros(num_cores)
    workload_per_core = np.zeros(num_cores)
    spike_count_per_core = np.zeros(num_cores)
    
    for idx, core in enumerate(cores):
        # Populations in this core
        pops_in_core = np.where(assignments == core)[0]
        population_counts[idx] = len(pops_in_core)
        
        # Sum intra-core functional coupling
        core_J = J[np.ix_(pops_in_core, pops_in_core)]
        if len(pops_in_core) > 0:
            intra_coupling[idx] = np.sum(core_J) / (len(pops_in_core)**2)
        else:
            intra_coupling[idx] = 0.0
        
        # Calculate workload: neuron updates + spikes
        n_neurons = sum(populations[p].N for p in pops_in_core)
        n_spikes = sum(len(neuron_spikes) for p in pops_in_core for neuron_spikes in populations[p].spike_times)

        workload_per_core[idx] = n_neurons * neuron_update_cost + n_spikes * spiking_cost
        
        spike_count_per_core[idx] = n_spikes
    
    # 3. Load Balance (Coefficient of Variation)
    workload_mean = np.mean(workload_per_core)
    workload_std = np.std(workload_per_core)
    load_balance = workload_std / workload_mean if workload_mean > 0 else 0.0
    
    # 5. Cross-Partition Interference (simplified)
    cross_partition_interference = 0.0
    if comm_bandwidth_matrix is not None:
        for i in range(num_cores):
            for j in range(num_cores):
                if i != j:
                    # Sum J between populations in core i and core j
                    pops_i = np.where(assignments == cores[i])[0]
                    pops_j = np.where(assignments == cores[j])[0]
                    inter_J = np.sum(J[np.ix_(pops_i, pops_j)])
                    bandwidth = comm_bandwidth_matrix[i, j]
                    if bandwidth > 0:
                        cross_partition_interference += inter_J / bandwidth
    
    # 6. Power/Energy Estimate (simple linear)
    # Power ~ frequency * power_per_freq_unit + communication power proportional to comm cost
    # For now, estimate frequency allocation below
    
    # Frequency allocation optimization (simple proportional allocation)
    # Allocate frequencies proportional to workload, clipped by min/max and normalized to total budget
    freq_alloc = workload_per_core.copy()
    total_workload = np.sum(workload_per_core)
    if total_workload > 0:
        freq_alloc = freq_alloc / total_workload * total_frequency_budget
    freq_alloc = np.clip(freq_alloc, f_min, f_max)
    
    # Compute time cost per core
    time_cost_per_core = workload_per_core / freq_alloc
    
    # Total simulation time is max core time
    total_sim_time = np.max(time_cost_per_core)
    
    # Power estimate
    power_compute = np.sum(freq_alloc) * power_per_freq_unit
    power_comm = comm_cost  # simplified: proportional to communication cost
    total_power = power_compute + power_comm
    
    # Resource Utilization Efficiency (workload / allocated frequency sum)
    resource_util_eff = total_workload / np.sum(freq_alloc) if np.sum(freq_alloc) > 0 else 0.0
    
    # Package all metrics
    metrics = {
        'communication_cost': comm_cost,
        'intra_partition_coherence': np.mean(intra_coupling),
        'load_balance_cv': load_balance,
        'cross_partition_interference': cross_partition_interference,
        'total_simulation_time': total_sim_time,
        'power_estimate': total_power,
        'resource_utilization_efficiency': resource_util_eff,
        'frequency_allocation': freq_alloc,
        'time_cost_per_core': time_cost_per_core,
        'workload_per_core': workload_per_core,
        'spike_count_per_core': spike_count_per_core,
    }
    
    return metrics
