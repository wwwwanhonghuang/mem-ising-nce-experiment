import numpy as np
from tqdm import tqdm
import os
import yaml

from pyclasses.ising_model import IsingModel, IsingModelIO
from pyclasses.ising_inferencer import IsingInferencer


def metropolis_sample(ising_model, ising_inferencer, num_steps, beta = 1):
    spin_configuration = np.random.choice([0, 1], size=ising_model.n_sites)

    for step in range(num_steps):
        # Choose a random spin
        i = np.random.randint(ising_model.n_sites)
        energy_before = inferencer.calculate_energy(ising_model, spin_configuration)
        energy_after = inferencer.calculate_energy(ising_model, spin_configuration)

        interaction_term = np.dot(ising_model.J[i], spin_configuration)

        if spin_configuration[i] == 1:
            # Flipping from 1 to 0
            delta_E = -2 * (interaction_term + ising_model.H[i])
        else:
            # Flipping from 0 to 1
            delta_E = 2 * (interaction_term + ising_model.H[i])
        
        # Accept or reject the move
        if np.random.rand() < np.exp(-beta * delta_E):
            spin_configuration[i] *= -1  # Flip spin
    
    return spin_configuration

def sample_n_configurations(ising_model, ising_inferencer, num_samples, num_steps, beta=1):
    """
    Sample N spin configurations using the Metropolis algorithm.

    Parameters:
        ising_model (object): An Ising model object containing system parameters.
        ising_inferencer (object): An inferencer object that computes energy.
        num_samples (int): Number of spin configurations to sample.
        num_steps (int): Number of Metropolis steps per sample.
        beta (float): Inverse temperature (1/kT).
    
    Returns:
        np.ndarray: An array of shape (num_samples, n_sites), where each row is a sampled configuration.
    """
    samples = []
    for _ in tqdm(range(num_samples)):
        sample = metropolis_sample(ising_model, ising_inferencer, num_steps, beta)
        samples.append(sample)
    
    return np.array(samples)


with open("3_ising_spin_configuration_sampling/config.yaml", "r") as file:
    config = yaml.safe_load(file)


ising_model_file = config["mcmc_sampling"]["ising_model_file"]
Z1 = config["mcmc_sampling"]["Z1"]
Z2 = config["mcmc_sampling"]["Z2"]
log_2_n_samples = config["mcmc_sampling"]["log_2_n_samples"]
log_2_sample_interval = config["mcmc_sampling"]["log_2_sample_interval"]
beta = config["mcmc_sampling"]["beta"]
output_file = config["mcmc_sampling"]["output_file"]

ising_model = IsingModelIO.read_ising_model_from_file(ising_model_file)
inferencer = IsingInferencer(Z1 = Z1, Z2 = Z2)
N = 1 << log_2_n_samples
num_steps = 1 << log_2_sample_interval
samples = sample_n_configurations(ising_model, inferencer, N, num_steps, beta)
np.save(output_file, samples)
