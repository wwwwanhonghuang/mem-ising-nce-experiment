import numpy as np

class IsingModel:
    def __init__(self, n_sites, H = None, J = None, temperature = 1):
        self.H = np.zeros((n_sites)) if H is None else H
        self.J = np.zeros((n_sites, n_sites)) if J is None else J
        self.temperature = temperature
        self.n_sites = n_sites


    
class IsingModelIO:
    def __init__(self):
        pass

    @classmethod
    def read_ising_model_from_file(cls, file_path):
        with open(file_path, 'r') as f:
            context = f.read().split("\n")
            f.close()
            n_sites = int(context[0])
            temperature = float(context[1])
            J = np.zeros((n_sites, n_sites))
            H = np.zeros((n_sites))
            
            for i in range(n_sites):
                parameters = context[2 + i].strip().split(" ")
                for j in range(n_sites):
                    J[i, j] = float(parameters[j])

            for i in range(n_sites):
                parameters = context[2 + n_sites].strip().split(" ")
                H[i] = float(parameters[i])
            return IsingModel(n_sites, H, J, temperature)
            
