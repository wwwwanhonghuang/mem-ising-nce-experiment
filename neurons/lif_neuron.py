import numpy as np
import matplotlib.pyplot as plt

class LIFNeuronPopulation:
    def __init__(self, N,
                 tau_m=20.0,
                 R=1.0,
                 V_th=-50.0,
                 V_reset=-65.0,
                 V_rest=-65.0,
                 dt=1.0,
                 W=None,
                 noise_std=0.5, weight_mean=1, weight_std=0.5):
        self.N = N
        self.tau_m = tau_m
        self.R = R
        self.V_th = V_th
        self.V_reset = V_reset
        self.V_rest = V_rest
        self.dt = dt
        self.noise_std = noise_std
        
        self.V = self.V_rest + np.random.uniform(-5, 5, size=N)
        self.spike_times = [[] for _ in range(N)]
        
        if W is None:
            self.W = np.random.normal(weight_mean, weight_std, size=(N, N))
        else:
            assert W.shape == (N, N), "Weight matrix must be NxN"
            self.W = W

    def step(self, I, t):
        dV = (-(self.V - self.V_rest) + self.R * I) / self.tau_m * self.dt
        noise = np.random.normal(0, self.noise_std, size=self.N)
        self.V += dV + noise
        
        spikes = self.V >= self.V_th
        for i, spiked in enumerate(spikes):
            if spiked:
                self.spike_times[i].append(t)
                self.V[i] = self.V_reset
        return spikes

# # Simulation parameters
# N = 8
# T = 150
# dt = 1.0
# time = np.arange(0, T, dt)

# W = np.random.normal(0, 0.05, (N, N))

# neuron_pop = LIFNeuronPopulation(N=N, W=W, dt=dt, noise_std=0.7)

# I_ext = np.zeros((len(time), N))
# base_current = 15
# noise_level = 0.4
# for n in range(N):
#     I_ext[30:100, n] = base_current + np.random.uniform(-noise_level, noise_level)

# V_trace = np.zeros((len(time), N))
# spikes_record = np.zeros((len(time), N), dtype=bool)

# S_prev = np.zeros(N)

# for t_i, t in enumerate(time):
#     I = I_ext[t_i] + W @ S_prev
#     spikes = neuron_pop.step(I, t)
#     V_trace[t_i] = neuron_pop.V
#     spikes_record[t_i] = spikes
#     S_prev = spikes.astype(float)
    

# fig, axes = plt.subplots(N, 1, figsize=(12, 2 * N), sharex=True)
# colors = plt.cm.get_cmap('tab10', N)

# for i in range(N):
#     ax = axes[i] if N > 1 else axes
#     ax.plot(time, V_trace[:, i], color=colors(i), label=f'Neuron {i+1}')
#     ax.axhline(neuron_pop.V_th, color='r', linestyle='--', alpha=0.7)
#     spike_times_i = np.array(neuron_pop.spike_times[i])
#     ax.vlines(spike_times_i, ymin=neuron_pop.V_reset, ymax=neuron_pop.V_th,
#               color='k', alpha=0.4)
#     ax.set_ylabel('V (mV)')
#     ax.legend(loc='upper right')
#     ax.grid(True)

# axes[-1].set_xlabel('Time (ms)')
# plt.suptitle('Membrane Potentials of Each Neuron (No Offset)')
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()


# # Plot membrane potentials with offsets
# fig, ax2 = plt.subplots(1, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1]}, sharex=True)
# colors = plt.cm.get_cmap('tab10', N)


# # Raster plot of spikes
# for neuron_idx in range(N):
#     spike_times_i = neuron_pop.spike_times[neuron_idx]
#     ax2.scatter(spike_times_i, np.full_like(spike_times_i, neuron_idx), color=colors(neuron_idx), s=20)
# ax2.set_yticks(range(N))
# ax2.set_yticklabels([f'Neuron {i+1}' for i in range(N)])
# ax2.set_xlabel('Time (ms)')
# ax2.set_ylabel('Neuron Index')
# ax2.set_title('Spike Raster Plot')
# ax2.grid(True)

# plt.tight_layout()
# plt.show()
