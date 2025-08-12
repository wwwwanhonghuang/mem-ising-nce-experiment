from current_generator.base_current_generator import BaseIntervalCurrentGenerator
import numpy as np
class IntervalCurrentGenerator(BaseIntervalCurrentGenerator):
    def __init__(self, num_pops, n_times, neurons_per_pop, start = 20, base_current = 12, noise_level = 0.4, zero_current_interval = 200, current_interval = 100):
        self.num_pops = num_pops
        self.n_times = n_times
        self.neurons_per_pop = neurons_per_pop
        self.start = start
        self.base_current = base_current
        self.noise_level = noise_level
        self.zero_current_interval = zero_current_interval
        self.current_interval = current_interval
        
    def generate_currents(self):
        # External input: shape (time, num_pops, neurons_per_pop)
        I_ext = np.zeros((self.n_times, self.num_pops, self.neurons_per_pop))

    
        # Example: provide a pulse input to each population at different time windows, add small variation per neuron
        for p in range(self.num_pops):
            if p > 1:
                break
            start = self.start
            base_current = self.base_current
            noise_level = self.noise_level
            zero_current_interval = self.zero_current_interval
            current_interval = self.current_interval
            
            for n in range(self.neurons_per_pop):
                current_time = start
                while current_time < self.n_times:
                    end = current_time + current_interval
                    I_ext[current_time: end, p, n] = base_current + np.random.uniform(-noise_level, noise_level)
                    current_time += zero_current_interval
                    
        return I_ext