import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.special import logsumexp

# class ConfigurationIterator:
#     def __init__(self, n_sites):
#         self.n_sites = n_sites
#         self.current_configuration = np.zeros(n_sites, dtype=int)
#         self.current_index = 0

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.current_index >= (1 << self.n_sites):
#             raise StopIteration
#         if self.current_index == 0:
#             # Return the initial configuration (all zeros)
#             self.current_index += 1
#             return self.current_configuration.copy()
#         else:
#             # Find the bit that changes between the current and previous index
#             changed_bit = int(np.log2(self.current_index ^ (self.current_index - 1)))
#             # Flip the bit in the configuration
#             self.current_configuration[changed_bit] = 1 - self.current_configuration[changed_bit]
#             self.current_index += 1
#             return self.current_configuration.copy()
def gray_code_flip_bit(i):
    g_i = i ^ (i >> 1)
    g_next = (i + 1) ^ ((i + 1) >> 1)
    flip_mask = g_i ^ g_next  # only one bit is set
    
    # Find index of the flipped bit (0-based)
    # For example, if flip_mask = 0b1000, flipped bit is 3
    flipped_bit = (flip_mask.bit_length() - 1)
    return flipped_bit

class ConfigurationIterator:
    def __init__(self, n_sites):
        self.n_sites = n_sites
        self.total = 1 << n_sites
        self.index = 0
        self.current = np.ones(n_sites, dtype=int)  # initialize all +1 spins
    
    def __iter__(self):
        self.index = 0
        self.current = np.ones(self.n_sites, dtype=int)
        return self
    
    def __next__(self):
        if self.index >= self.total:
            raise StopIteration
        if self.index == 0:
            self.index += 1
            return self.current.copy(), None  # no flip at first config
        flipped_bit = gray_code_flip_bit(self.index - 1)
        self.current[flipped_bit] *= -1  # flip spin at flipped_bit
        self.index += 1
        return self.current.copy(), flipped_bit


class ConfigurationGenerator():
    @classmethod
    def all_configurations(cls, n_sites):
        n_conf = 1 << n_sites  # total number of configurations
        configs = np.zeros((n_conf, n_sites), dtype=np.uint8)
        for bit in range(n_sites):
            # Each column alternates 0/1 in blocks of size 2**bit
            block_size = 1 << bit
            pattern = np.tile(np.concatenate([np.zeros(block_size, dtype=np.uint8),
                                            np.ones(block_size, dtype=np.uint8)]),
                            n_conf // (block_size * 2))
            configs[:, bit] = pattern
        return configs


class PairwiseIsingModel(object):
    def __init__(self, n_sites):
        self.n_sites = n_sites
        self.J = np.zeros((n_sites, n_sites))  # Coupling matrix
        self.H = np.zeros(n_sites)  # External field
_cached_configs = None     
class PairwiseIsingModelInferencer:
    def __init__(self, ising_model: PairwiseIsingModel):
        self.Z = 0  # Partition function
        self.ising_model = ising_model
        self.logZ = -np.inf
        self.beta = 1
    
    def _check_partition_function(self):
        """Ensure the partition function is computed."""
        if self.Z == 0:
            self.update_partition_function()
    
    # def update_partition_function(self):
    #     """Update the partition function Z by iteratively generating configurations using Gray codes."""
    #     self.Z = 0
    #     config_iterator = ConfigurationIterator(self.ising_model.n_sites)
    #     for configuration in config_iterator:
    #         self.Z += np.exp(-self.energy(configuration))
    
    # def update_partition_function(self, n_jobs=-1, configs = None):
    #     """Parallel update of partition function Z."""
    #     if configs is None:
    #         configs = list(ConfigurationIterator(self.ising_model.n_sites))
        
    #     def contrib(config):
    #         return np.exp(-self.energy(config))
        
    #     results = Parallel(n_jobs=n_jobs)(delayed(contrib)(cfg) for cfg in configs)
    #     self.Z = np.sum(results)
        

    def update_partition_function(self, configs=None):
        n = self.ising_model.n_sites
        total_configs = 1 << n
        
        config_iter = ConfigurationIterator(n)
        
        energies = []
        energy = None
        
        for i, (config, flipped_bit) in enumerate(tqdm(config_iter, total=total_configs, desc="Partition function")):
            if flipped_bit is None:
                energy = self.energy(config)
            else:
                spin_val_before_flip = -config[flipped_bit]
                row_sum = np.dot(self.ising_model.J[flipped_bit, :], config)
                col_sum = np.dot(config, self.ising_model.J[:, flipped_bit])
                delta_E = 2 * spin_val_before_flip * (row_sum + col_sum + self.ising_model.H[flipped_bit])
                energy += delta_E
            
            energies.append(-energy)  # store -energy for log-sum-exp
        
        energies = np.array(energies)
        m = np.max(energies)
        Z_log = m + np.log(np.sum(np.exp(energies - m)))
        print(f'Z_log = {Z_log}')
        self.logZ = Z_log
        self.Z = np.exp(Z_log)

    def energy(self, configuration):
        """Compute the energy of a given configuration."""
        configuration = np.array(configuration)
        return -np.sum(configuration @ self.ising_model.J @ configuration.T) - np.dot(self.ising_model.H, configuration)

    def energies_all(self, configs):
        J = self.ising_model.J
        H = self.ising_model.H
        
        # configs: (num_configs, n) with spins ±1
        quadratic = np.einsum('ij,jk,ik->i', configs, J, configs)  # shape (num_configs,)
        linear = configs @ H  # shape (num_configs,)
        energies = -(quadratic + linear)
        return energies

    def probability(self, configuration):
        """Compute the probability of a given configuration."""
        if self.Z == 0:
            print("Warning: Ising model's partition function Z == 0. The partition function may not have been initialized yet.")
            return 0
        return np.exp(-self.energy(configuration)) / self.Z


    def log_probability(self, configuration):
        """Compute the probability of a given configuration."""
        if self.Z == 0:
            print("Warning: Ising model's partition function Z == 0. The partition function may not have been initialized yet.")
            return 0
        return -self.energy(configuration) - self.logZ
    
    def essembly_average_si(self):
        n = self.ising_model.n_sites
        configs = ConfigurationGenerator.all_configurations(n).astype(np.int8)
        spins = (2 * configs - 1).astype(np.float64)  # ±1 spins

        energies = self.energies(spins)
        log_weights = -self.beta * energies

        # Z in log space
        log_Z = logsumexp(log_weights)

        # Numerator: weighted sum of spins
        weights = np.exp(log_weights - log_Z)  # normalized probabilities
        avg_si = np.sum(weights[:, None] * spins, axis=0)

        return avg_si
    
    
    def energies(self, configurations):
        """
        Compute energies for multiple configurations at once.

        Parameters:
            configurations: np.ndarray of shape (num_configs, n_sites),
                            with spins ±1.

        Returns:
            np.ndarray of shape (num_configs,), energies of each configuration.
        """
        # Ensure numpy array
        configs = np.array(configurations, dtype=float)

        # Interaction term: - s^T J s for each configuration
        # This is a quadratic form for each row:
        # Use einsum or batch matrix multiplication
        interaction_energies = -np.einsum('bi,ij,bj->b', configs, self.ising_model.J, configs)

        # Field term: - H^T s for each configuration
        field_energies = -configs @ self.ising_model.H

        # Total energy
        total_energies = interaction_energies + field_energies

        return total_energies

    def essembly_average_sisj(self):
        """Compute the ensemble average <s_i s_j> for all pairs of spins."""
        self._check_partition_function()  # Ensure the partition function is computed
        n = self.ising_model.n_sites
        configs = _cached_configs if _cached_configs is not None else ConfigurationGenerator.all_configurations(n)  # (2^n, n), spins 0/1 or ±1

        energies = self.energies(configs)
        log_unnormalized = -self.beta * energies

        m = np.max(log_unnormalized)
        weights = np.exp(log_unnormalized - m)  # stable weights

        weighted_configs = configs.T * weights  # shape (n, num_configs)
        ensemble_average_sisj = weighted_configs @ configs  # shape (n, n)

        log_Z = logsumexp(log_unnormalized)
        ensemble_average_sisj /= np.exp(log_Z - m)  # normalize
        print(ensemble_average_sisj)
        return ensemble_average_sisj
        
    
    # def essembly_average_si(self):
    #     n = self.ising_model.n_sites
    #     configs = _cached_configs if _cached_configs is not None else ConfigurationGenerator.all_configurations(n)  # (2^n, n), spins 0/1 or ±1
        
    #     # If needed, convert 0/1 to ±1
    #     configs = 2 * configs - 1

    #     # Compute energies vectorized for all configs
    #     energies = self.energies_all(configs)  # shape (2^n,)
        
    #     # Compute log probabilities: log P(c) = -beta * E(c) - log Z
    #     log_unnormalized = -self.beta * energies
        
    #     # Compute log partition function log Z safely
    #     log_Z = logsumexp(log_unnormalized)
        
    #     # Normalize log probabilities
    #     log_probs = log_unnormalized - log_Z
        
    #     # Convert to probabilities safely (avoid overflow)
    #     probs = np.exp(log_probs)  # sum to 1

    #     # Compute ensemble averages
    #     avg_si = np.sum(probs[:, None] * configs, axis=0)  # shape (n,)

    #     return avg_si
        
    
    # def essembly_average_sisj(self):
    #     n = self.ising_model.n_sites
    #     configs = _cached_configs if _cached_configs is not None else ConfigurationGenerator.all_configurations(n)  # (2^n, n), spins 0/1 or ±1
    #     configs = 2 * configs - 1  # convert to ±1 spins

    #     energies = self.energies_all(configs)
    #     print(energies.shape)

    #     log_unnormalized = -self.beta * energies
    #     log_Z = logsumexp(log_unnormalized)
    #     log_probs = log_unnormalized - log_Z
    #     probs = np.exp(log_probs)

    #     # Compute <s_i s_j> as weighted sum of outer products
    #     # Efficiently: weighted dot product of configurations
    #     weighted_configs = configs.T * probs  # shape (n, 2^n)
    #     avg_sisj = weighted_configs @ configs  # shape (n, n)

    #     return avg_sisj