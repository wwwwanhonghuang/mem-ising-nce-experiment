
import numpy as np
from tqdm import tqdm

class HardwareConfiguration(object):
    def __init__(self, n_cores, n_chips, frequency_configuration):
        self.n_cores = n_cores
        self.n_chips = n_chips
        self.frequency_configuration = frequency_configuration
        
        
        
        

class Network():
    def __init__(self, num_pops, neurons_per_pop, connection_matrices, populations):
        self.num_pops = num_pops
        self.neurons_per_pop = neurons_per_pop
        self.connection_matrices = connection_matrices
        self.populations = populations

COST_KEYS = ['total_cost', 'input_intergation_cost', 'external_input_cost', 'spiking_cost']

class PerformanceMonitor():
    
    def default_value(self, key):
            return 0
        
    def __init__(self):
        self.cost = {cost_key: self.default_value(cost_key) for cost_key in COST_KEYS}
        
        
    def initialize(self, hardware_configuration):
        self.cost = {cost_key: self.default_value(cost_key) for cost_key in COST_KEYS}
        
    def step_simulation_initialization_cost(self, hardware_configuration):
        self.cost['total_cost'] += 100

    def step_population_input_intergation_cost(self, hardware_configuration, deployment_configuration, source_pop, target_pop, S_prev, t_i, t):
        N_source = S_prev[source_pop].size
        N_target = S_prev[target_pop].size
        self.cost['total_cost'] += N_source * N_target
        self.cost['input_intergation_cost'] += N_source * N_target

    def step_process_external_input_cost(self, hardware_configuration, deployment_configuration, inputs, I_ext, t_i, t):
        self.cost['total_cost'] += np.prod(inputs.shape)
        self.cost['external_input_cost'] += np.prod(inputs.shape)

    def step_spiking_cost(self, hardware_configuration, deployment_configuration, spikes, t_i, t):
        self.cost['total_cost'] += np.sum(spikes)
        self.cost['spiking_cost'] += np.sum(spikes)


    def report_performance(self):
        print(f"Report performance profiled...")
        for key in self.cost:
            print(f'\t{key} Cost == {self.cost[key]}')
        print()
    
class VirtualNeuromorphicHardware():
    def __init__(self, configuration: HardwareConfiguration = None):
        self.configuration = configuration
        self.performance_monitor : PerformanceMonitor = PerformanceMonitor()
        
        
    def set_configuration(self, configuration: HardwareConfiguration):
        self.configuration = configuration
                
    def do_simulation(self, n_simulation_times, dt, deployment_configuration, network: Network, I_ext):
        if self.configuration is None:
            return None
        
        hardware_configuration = self.configuration
        performance_monitor = self.performance_monitor
        
        time = np.arange(0, n_simulation_times, dt)


        V_trace = np.zeros((len(time), network.num_pops, network.neurons_per_pop))
        spikes_record = np.zeros((len(time), network.num_pops, network.neurons_per_pop), dtype=bool)

        performance_monitor.initialize(hardware_configuration)
        

        # Initialize spike state for previous step (per population)
        S_prev = np.zeros((network.num_pops, network.neurons_per_pop))
        performance_monitor.step_simulation_initialization_cost(hardware_configuration=hardware_configuration)

        for t_i, t in tqdm(enumerate(time)):
            # Compute inputs to each population from all populations
            inputs = np.zeros((network.num_pops, network.neurons_per_pop))
            for target_pop in range(network.num_pops):
                # Sum over all source populations
                for source_pop in range(network.num_pops):
                    inputs[target_pop] += network.connection_matrices[target_pop, source_pop] @ S_prev[source_pop]
                    performance_monitor.step_population_input_intergation_cost(hardware_configuration, deployment_configuration, source_pop, target_pop, S_prev, t_i, t)
                    
            # Add external input
            total_input = inputs + I_ext[t_i]
            performance_monitor.step_process_external_input_cost(hardware_configuration, deployment_configuration, inputs, I_ext, t_i, t)

            # Step populations independently
            for p in range(network.num_pops):
                spikes = network.populations[p].step(total_input[p], t)
                V_trace[t_i, p] = network.populations[p].V
                spikes_record[t_i, p] = spikes
                S_prev[p] = spikes
                performance_monitor.step_spiking_cost(hardware_configuration, deployment_configuration, spikes, t_i, t)

        spike_times_all = []
        neuron_indices_all = []

        for pop_idx in range(network.num_pops):
            for neuron_idx in range(network.neurons_per_pop):
                # Get spike times for this neuron
                stimes = network.populations[pop_idx].spike_times[neuron_idx]
                # Global neuron ID (pop * neurons_per_pop + neuron)
                global_id = pop_idx * network.neurons_per_pop + neuron_idx
                spike_times_all.extend(stimes)
                neuron_indices_all.extend([global_id] * len(stimes))

        spike_times_all = np.array(spike_times_all)
        neuron_indices_all = np.array(neuron_indices_all)
        return spike_times_all, neuron_indices_all