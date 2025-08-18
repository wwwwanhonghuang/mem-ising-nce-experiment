from current_generator.base_current_generator import BaseIntervalCurrentGenerator
import numpy as np
class IntervalCurrentGenerator(BaseIntervalCurrentGenerator):
    def __init__(self, num_pops, n_times, neurons_per_pop, start = 20, base_current = 12, noise_level = 0.4, zero_current_interval = 200, current_interval = 100, select_neurons=None, limit_current_times = None):
        self.num_pops = num_pops
        self.n_times = n_times
        self.neurons_per_pop = neurons_per_pop
        self.start = start
        self.base_current = base_current
        self.noise_level = noise_level
        self.zero_current_interval = zero_current_interval
        self.current_interval = current_interval
        self.select_neurons = select_neurons
        self.limit_current_times = limit_current_times
        
    def generate_currents(self):
        # External input: shape (time, num_pops, neurons_per_pop)
        I_ext = np.zeros((self.n_times, self.num_pops, self.neurons_per_pop))

        start = self.start
        current_time = start
        apply_current_times = 0
        base_current = self.base_current
        noise_level = self.noise_level
        zero_current_interval = self.zero_current_interval
        current_interval = self.current_interval
        
                
        # Provide current for 'current_interval', 
        # then stop current for 'zero_current_interval', and repeat. Maximum repeat times of a neuron population = limit_current_times_this_population
        while current_time < self.n_times:
            
            for p_index in (range(self.num_pops) if (self.select_neurons is None) else range(len(self.select_neurons))):
                selected_neuron = self.select_neurons[p_index] if self.select_neurons is not None else p_index
                limit_current_times_this_population = self.limit_current_times[selected_neuron] if self.limit_current_times is not None else np.inf
                if apply_current_times >= limit_current_times_this_population:
                    continue
                                
                for n in (range(self.neurons_per_pop)):    
                    end = current_time + current_interval
                    I_ext[current_time: end, selected_neuron, n] = base_current + np.random.uniform(-noise_level, noise_level)
            apply_current_times += 1
            current_time += current_interval + zero_current_interval
        return I_ext