import argparse
import yaml
import os
import re
import pickle
from ising_models.ising_trainer import TrainingContext, PairwiseIsingModel

import numpy as np
from typing import List, Tuple, Callable, Optional

parser = argparse.ArgumentParser()
parser.add_argument("--configuration-file-path", type=str)

args = parser.parse_args()


# Logics for loading  the best ising model's parameters J

configuration_file_path = args.configuration_file_path
print(f'Use configuration file: {configuration_file_path}')

# Load YAML config
with open(configuration_file_path, 'r') as file:
    config = yaml.safe_load(file)

print("loaded configurations:")
print(config)

n_sites = config['network']['n_pops']
project_name = config['project']['name']

images_root = os.path.join("projects", project_name, "images")
model_save_root = os.path.join("projects", project_name, "model_save")
data_root = os.path.join("projects", project_name, "data")
image_save_path_root = images_root

n_epoches = config['ising_training']['n_epoches']
neurons_per_pop = config['network']['neurons_per_pop']
record_file_path = os.path.join(data_root, config['network']['spike_data_store']['path'])
model_save_path_root = model_save_root


files_in_model_store_root = os.listdir(model_save_path_root)
file_names_and_loss = [(file_name, float(re.match("epoch_([0-9])+_loss_([0-9]+\\.[0-9]*)\\.pkl", file_name).group(2))) for file_name in files_in_model_store_root]

sorted(file_names_and_loss, key=lambda x: x[1])
assert len(file_names_and_loss) > 0, "Cannot find avaliable pkl file in model save root"

best_model_file = os.path.join(model_save_path_root, file_names_and_loss[0][0])
print(f'best model = {best_model_file}')


with open(best_model_file, 'rb') as f:
    context : TrainingContext = pickle.load(f)
    
assert context is not None, "Cannot load context"

model: PairwiseIsingModel = context.model

J = model.J

n_sites = J.shape[0]

# Partitioning

from partitioning_algorithms.utils import *

# ---------- High-level API ----------

def partition_J(
    J: np.ndarray, 
    K: int, 
    method: str = "greedy", 
    use_abs: bool = True, 
    refine: bool = True
):
    """
    Partition sites into groups of size <= K using various methods.
    
    Supported methods:
      - 'greedy'
      - 'agglomerative'
      - 'spectral'
      - 'simple_kmedoids'   : k-medoids like clustering using only numpy/scipy
      - 'louvain' : community detection using NetworkX greedy modularity - louvain
      - 'kernighan_lin'     : recursive Kernighan–Lin bisection
    
    Returns:
      parts: List of lists of site indices
      score: Intra-group weight sum
    """
    
    methods: dict[str, Callable[..., List[List[int]]]] = {
        "greedy": partition_greedy,
        "agglomerative": partition_agglomerative,
        "spectral": partition_spectral,
        "simple_kmedoids": partition_simple_kmedoids,
        "louvain": partition_louvain,
        "kernighan_lin": partition_kernighan_lin,
    }

    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(methods.keys())}.")

    # Partition
    parts = methods[method](J, K, use_abs=use_abs)

    # Optional local refinement
    if refine:
        parts = refine_local_swaps(J, parts, use_abs=use_abs, max_passes=10)

    # Compute intra-group score
    score = intra_score(J, parts, use_abs=use_abs)

    return parts, score


methods = ['greedy', 'agglomerative', 'spectral', 'simple_kmedoids', 'louvain', 'kernighan_lin']
partition_save_root = os.path.join("projects", project_name, "partition_schemes")
os.makedirs(partition_save_root, exist_ok=True)
 
for method in methods:
    print(f'use_{method} for partitioning')
    parts, score = partition_J(J, K=4, method=method, use_abs=True, refine=True)
    print(parts, score)
    np.save(os.path.join(partition_save_root, f"partition_scheme_core_only_{method}.npy"), np.asarray([parts, score], dtype=object), allow_pickle=True)
    
