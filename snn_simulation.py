import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(self, 
                 tau_m=20.0,    # membrane time constant (ms)
                 R=1.0,         # membrane resistance (MΩ)
                 V_th=-50.0,    # spike threshold (mV)
                 V_reset=-65.0, # reset potential (mV)
                 V_rest=-65.0,  # resting potential (mV)
                 dt=1.0):       # time step (ms)
        self.tau_m = tau_m
        self.R = R
        self.V_th = V_th
        self.V_reset = V_reset
        self.V_rest = V_rest
        self.dt = dt
        self.V = V_rest
        self.spike_times = []

    def step(self, I, t):
        # Update membrane potential
        dV = (-(self.V - self.V_rest) + self.R * I) / self.tau_m * self.dt
        self.V += dV

        # Check for spike
        if self.V >= self.V_th:
            self.spike_times.append(t)
            self.V = self.V_reset  # reset voltage after spike
            return True
        return False

# Simulation parameters
T = 200        # total time (ms)
dt = 1.0       # time step (ms)
time = np.arange(0, T, dt)

# Input current: constant current pulse between 50ms and 150ms
I = np.zeros(len(time))
I[(time >= 50) & (time <= 150)] = 1.5  # nA

# Create LIF neuron
neuron = LIFNeuron(dt=dt)

# Record membrane potential
V_trace = []

# Run simulation
for t_i, t in enumerate(time):
    neuron.step(I[t_i], t)
    V_trace.append(neuron.V)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time, V_trace, label='Membrane potential (mV)')
plt.plot(time, I*20 - 80, 'r--', label='Input current (scaled)')
plt.axhline(neuron.V_th, color='k', linestyle=':', label='Threshold')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.title('LIF Neuron Simulation')
plt.legend()
plt.show()
