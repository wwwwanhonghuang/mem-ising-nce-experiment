import numpy as np

class IsingInferencer:
    def __init__(self, Z1 = 0, Z2 = 0):
        self.Z1 = Z1
        self.Z2 = Z2

    def calculate_energy(self, ising_model, spin_configuration, order = 2):
        energy = 0
        if order != 1 and order != 2:
            raise ValueError
        if order == 2:
            energy += -np.sum(ising_model.J * np.outer(spin_configuration, spin_configuration))
        energy += -np.dot(ising_model.H, spin_configuration)
        return energy
        