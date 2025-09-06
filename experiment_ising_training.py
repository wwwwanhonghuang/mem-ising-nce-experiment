import numpy as np
import matplotlib.pyplot as plt
from ising_models.ising_model import PairwiseIsingModelInferencer, PairwiseIsingModel, ConfigurationIterator, ConfigurationGenerator
import argparse
import os
from ising_models.ising_trainer import PairwiseIsingModelTrainer
from tqdm import tqdm
import pickle
import time
import yaml
from numpy.lib.stride_tricks import sliding_window_view

import matplotlib.pyplot as plt
import numpy as np

def sliding_window_mean(mat, K):
    windows = sliding_window_view(mat, (K, mat.shape[1]))[:, 0, :, :]
    return windows.mean(axis=1)

parser = argparse.ArgumentParser()
parser.add_argument('--plus_n_std', type=int, default=0.5)
parser.add_argument('--configuration-file-path', type=str)

window_size = 10

args = parser.parse_args()

configuration_file_path = args.configuration_file_path
print(f'Use configuration file: {configuration_file_path}')

# Load YAML config
with open(configuration_file_path, 'r') as file:
    config = yaml.safe_load(file)

print("loaded configurations:")
print(config)

plus_n_std = args.plus_n_std
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




data = np.load(record_file_path)

neuron_indices_all = data['neuron_indices_all']
neuron_indices_all = [(global_id // neurons_per_pop, global_id % neurons_per_pop) for global_id in neuron_indices_all]

spike_times_all = data['spike_times_all']

T = config['network']['simulation_times']

spiking_mat = np.zeros((T, n_sites, neurons_per_pop))

for index, neural_idx in enumerate(neuron_indices_all):
    spiking_mat[int(spike_times_all[index]), neural_idx[0], neural_idx[1]] = 1
    
reduced_mat = spiking_mat.sum(axis=2) / neurons_per_pop

reduced_mat = sliding_window_mean(reduced_mat, window_size)

threshold = reduced_mat.flatten().mean() + plus_n_std * reduced_mat.flatten().std()

reduced_mat[reduced_mat < threshold] = 0
reduced_mat[reduced_mat >= threshold] = 1

plt.matshow(reduced_mat)

ising_activation_img_save_path = os.path.join(image_save_path_root, 'ising-binary-observation-data.png')
print(f'Saving observed ising activation map to {ising_activation_img_save_path}')

plt.savefig(ising_activation_img_save_path)

n_sites = n_sites
ising_model = PairwiseIsingModel(n_sites)
ising_model.J = np.random.rand(n_sites, n_sites)
ising_model.H = np.random.rand(n_sites)

print(f'Generate Configurations.')
configs = ConfigurationGenerator().all_configurations(n_sites=n_sites)
save_full_configuration = False
if save_full_configuration:
    np.save(f"data/full_configurations_n{n_sites}.npy", configs)
    

print(f'Initialize ising model. Calculate initial partition function Z...')
time_start = time.time()

inferencer = PairwiseIsingModelInferencer(ising_model)
inferencer.update_partition_function(configs=configs)

time_end = time.time()

duration = time_end - time_start
print("Time: {duration:.2f}s")

total_epoches = n_epoches
trainer = PairwiseIsingModelTrainer(ising_model, inferencer)
loss_record = []
pbar = tqdm(total=total_epoches, desc="Training", unit="epoch")

training_start_time = time.time()

global last_epoch_time

last_epoch_time = training_start_time

if not os.path.exists(model_save_path_root):
    os.makedirs(model_save_path_root, exist_ok=True)
def on_epoch_end(ctx):
    global last_epoch_time

    epoch = ctx.epoch
    epochs = ctx.epochs
    loss = ctx.loss
    kl_loss = ctx.kl_loss
    l2_loss = ctx.l2_loss

    now = time.time()
    epoch_duration = now - last_epoch_time
    last_epoch_time = now
    
    pbar.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, KL: {kl_loss:.4f}, L2: {l2_loss:.4f}")
    pbar.update(1) 
    print(f"Time: {epoch_duration:.2f}s")
    loss_record.append(loss)
    store_model_file = os.path.join(model_save_path_root, f"epoch_{ctx.epoch}_loss_{ctx.loss}.pkl")
    print(f'model save to {store_model_file}')
    with open(store_model_file, 'wb') as f:
        pickle.dump(ctx, f)

trained_model = trainer.train(reduced_mat, epochs=total_epoches, learning_rate=0.1, epoch_callback=on_epoch_end, configs=configs)



def plot_alignment(observation_dataset, inferencer):
    """Plot alignment between observation averages and model averages."""
    n_sites = observation_dataset.shape[1]
    
    # Compute observation averages
    essembly_average_obs_sisj = np.einsum('ki,kj->ij', observation_dataset, observation_dataset) / observation_dataset.shape[0]
    essembly_average_obs_si = np.mean(observation_dataset, axis=0)

    # Compute model averages
    essembly_average_model_sisj = inferencer.essembly_average_sisj()
    essembly_average_model_si = inferencer.essembly_average_si()

    # Flatten the pairwise correlation matrices for plotting
    obs_sisj_flat = essembly_average_obs_sisj.flatten()
    model_sisj_flat = essembly_average_model_sisj.flatten()

    # Plot pairwise correlations
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(obs_sisj_flat, model_sisj_flat, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # y = x line
    plt.xlabel("Observation $\langle s_i s_j \\rangle_{\\text{obs}}$")
    plt.ylabel("Model $\langle s_i s_j \\rangle_{\\text{model}}$")
    plt.title("Pairwise Correlations Alignment")
    plt.xlim([0.,1.])
    plt.ylim([0.,1.])
    
    # Plot single-site averages
    plt.subplot(1, 2, 2)
    plt.scatter(essembly_average_obs_si, essembly_average_model_si, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # y = x line
    plt.xlabel("Observation $\langle s_i \\rangle_{\\text{obs}}$")
    plt.ylabel("Model $\langle s_i \\rangle_{\\text{model}}$")
    plt.title("Single-Site Averages Alignment")

    plt.tight_layout()
    plt.xlim([0.,1.])
    plt.savefig(os.path.join(image_save_path_root, "compare.png"))
trainer.inferencer.update_partition_function()
plot_alignment(reduced_mat, trainer.inferencer)
