
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from collections import Counter
from math import log, lgamma
from scipy.special import gammaln

from collections import Counter
import zlib
from sklearn.neighbors import NearestNeighbors

class HardwareConfiguration(object):
    def __init__(self, n_cores, n_chips, frequency_configuration):
        self.n_cores = n_cores
        self.n_chips = n_chips
        self.frequency_configuration = frequency_configuration
        
        

class Network():
    def __init__(self, num_pops, neurons_per_pop, connection_matrices, populations):
        self.num_pops = num_pops
        self.neurons_per_pop = neurons_per_pop
        self.connection_matrices = connection_matrices
        self.populations = populations

COST_KEYS = ['total_cost', 'input_intergation_cost', 'external_input_cost', 'spiking_cost']

class PerformanceMonitor():
    
    def default_value(self, key):
            return 0
        
    def __init__(self):
        self.cost = {cost_key: self.default_value(cost_key) for cost_key in COST_KEYS}
        self.evaluated_entropy = None
        self.evaluated_compressions = None
        
    def initialize(self, hardware_configuration):
        self.cost = {cost_key: self.default_value(cost_key) for cost_key in COST_KEYS}
        self.evaluated_entropy = None
        self.evaluated_compressions = None
        
    def step_simulation_initialization_cost(self, hardware_configuration):
        self.cost['total_cost'] += 100

    def step_population_input_intergation_cost(self, hardware_configuration, deployment_configuration, source_pop, target_pop, S_prev, t_i, t):
        N_source = S_prev[source_pop].size
        N_target = S_prev[target_pop].size
        self.cost['total_cost'] += N_source * N_target
        self.cost['input_intergation_cost'] += N_source * N_target

    def step_process_external_input_cost(self, hardware_configuration, deployment_configuration, inputs, I_ext, t_i, t):
        self.cost['total_cost'] += np.prod(inputs.shape)
        self.cost['external_input_cost'] += np.prod(inputs.shape)

    def step_spiking_cost(self, hardware_configuration, deployment_configuration, spikes, t_i, t):
        self.cost['total_cost'] += np.sum(spikes)
        self.cost['spiking_cost'] += np.sum(spikes)


    def report_performance(self):
        print(f"Report performance profiled...")
        for key in self.cost:
            print(f'\t{key} Cost == {self.cost[key]}')
        print()
        
    def evaluate_partition_entropy(self):
        self.entropy = {}
        
    
from sklearn.neighbors import NearestNeighbors

# ---- helper functions ----
def _safe_div(a, b, eps=1e-12):
    return a / (b + eps)

def _entropy_from_probs(probs: np.ndarray, base: int = 2):
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    logs = np.log2(probs) if base == 2 else np.log(probs)
    return float(-np.sum(probs * logs))

def _binary_entropy(p: np.ndarray, base: int = 2):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    if base == 2:
        return -(p*np.log2(p) + (1-p)*np.log2(1-p))
    else:
        return -(p*np.log(p) + (1-p)*np.log(1-p))

def _binomial_entropy_exact(K, p_array, base=2):
    """
    Compute the exact binomial entropy per timestep for a population of K neurons.

    Parameters
    ----------
    K : int
        Number of neurons in the population.
    p_array : np.ndarray, shape (T,)
        Fraction of neurons spiking per timestep (0 <= p <= 1).
    base : float
        Logarithm base for entropy calculation.

    Returns
    -------
    entropy : np.ndarray, shape (T,)
        Binomial entropy for each timestep.
    """
    T = len(p_array)
    k_vals = np.arange(0, K+1)[:, None]  # shape (K+1, 1)
    p_vals = p_array[None, :]            # shape (1, T)
    
    p_vals = np.clip(p_vals, 1e-24, 1-1e-24)
    logC = gammaln(K+1) - gammaln(k_vals) - gammaln(K - k_vals + 1)  # shape (K+1, 1)

    # Compute unnormalized probabilities in log-space for stability
    log_probs = logC + k_vals*np.log(p_vals) + (K - k_vals)*np.log(1 - p_vals)  # shape (K+1, T)

    # Exponentiate and normalize safely
    max_log = np.max(log_probs, axis=0, keepdims=True)  # for numerical stability
    probs = np.exp(log_probs - max_log)
    probs_sum = probs.sum(axis=0, keepdims=True)
    probs /= np.clip(probs_sum, 1e-24, None)  # avoid division by zero

    # Compute entropy per timestep safely
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)) / np.log(base), axis=0)  # shape (T,)

    return entropy

def _skewness(x: np.ndarray):
    m = x.mean()
    s = x.std(ddof=0)
    if s <= 0: 
        return 0.0
    return float(((x - m)**3).mean() / (s**3))

def _excess_kurtosis(x: np.ndarray):
    m = x.mean()
    s = x.std(ddof=0)
    if s <= 0:
        return -3.0  # degenerate -> excess kurtosis -3 (all mass at mean)
    return float(((x - m)**4).mean() / (s**4) - 3.0)

def _lag1_autocorr(x: np.ndarray):
    # Pearson correlation between x[:-1] and x[1:]
    if x.size < 2:
        return 0.0
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:]  - x[1:].mean()
    denom = np.sqrt((x0**2).sum() * (x1**2).sum())
    if denom == 0:
        return 0.0
    return float((x0 * x1).sum() / denom)

def _burstiness(x: np.ndarray):
    # Goh & Barabasi burstiness measure B = (sigma - mu) / (sigma + mu)
    # where x is inter-event intervals; for counts-per-bin we compute intervals between active bins
    events = np.nonzero(x)[0]
    if events.size <= 1:
        return 0.0
    intervals = np.diff(events)
    mu = intervals.mean()
    sigma = intervals.std(ddof=0)
    if sigma + mu == 0:
        return 0.0
    return float((sigma - mu) / (sigma + mu))


def _miller_madow_correction(H_emp, m, T):
    """Add Miller–Madow correction for undersampled discrete distributions."""
    return H_emp + (m - 1) / (2 * T)

def _knn_entropy_estimator(X, k=3, base=2):
    """
    kNN entropy estimator for binary vectors.
    X: array of shape (T, K) with binary spike patterns.
    """
    T, K = X.shape
    Xf = X.astype(float)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(Xf)
    distances, _ = nbrs.kneighbors(Xf)
    # distances[:,0] = 0 (self), so take distances[:,k]
    r = distances[:, k]
    c_d = np.log(2)  # volume for binary vectors, simple approx
    H = K * np.log(2) - np.mean(np.log(r + 1e-12)) * (base/np.log(2))  # rough
    return float(H)

def binary_entropy(p, base=2):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -(p*np.log(p) + (1-p)*np.log(1-p)) / np.log(base)

def spike_count_entropy(flat, base=2):
    # sum spikes per row
    counts = flat.sum(axis=1)
    counts = counts.astype(int)  # convert to integer
    
    # compute histogram
    probs = np.bincount(counts, minlength=flat.shape[1]+1) / len(counts)
    
    # remove zeros
    probs = probs[probs > 0]
    
    # entropy
    return -np.sum(probs * np.log(probs) / np.log(base))

def independent_entropy(flat, base=2):
    freqs = flat.mean(axis=0)
    return np.sum(binary_entropy(freqs, base=base))

def knn_entropy(flat, k=5, base=2):
    # Treat each row as point in {0,1}^K, estimate entropy via distances
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='hamming').fit(flat)
    distances, _ = nbrs.kneighbors(flat)
    # remove self-distance (0th column)
    avg_log_dist = np.mean(np.log(distances[:,1]+1e-12))
    K = flat.shape[1]
    return K * np.log(2) - K * avg_log_dist / np.log(base)

def encode_sparse_ids(flat):
    """Encode neuron IDs per timestep as int32 list"""
    ids_list = []
    for row in flat:
        fired = np.where(row)[0]
        for neuron_id in fired:
            ids_list.append(int(neuron_id).to_bytes(4,'big'))
    raw_bytes = b''.join(ids_list)
    compressed = zlib.compress(raw_bytes)
    return len(raw_bytes), len(compressed)

def encode_run_length(flat):
    """Contiguous run encoding along neuron IDs"""
    bytes_list = []
    for row in flat:
        fired = np.where(row)[0]
        if len(fired)==0:
            continue
        start = fired[0]
        count = 1
        for i in range(1,len(fired)):
            if fired[i]==fired[i-1]+1:
                count += 1
            else:
                bytes_list.append(int(start).to_bytes(4,'big') + int(count).to_bytes(4,'big'))
                start = fired[i]
                count = 1
        bytes_list.append(int(start).to_bytes(4,'big') + int(count).to_bytes(4,'big'))
    raw_bytes = b''.join(bytes_list)
    compressed = zlib.compress(raw_bytes)
    return len(raw_bytes), len(compressed)

def evaluate_partition_compression_large_K(flat, sample_frac=0.1, K=512):
    T,K = flat.shape
    stats = {}

    # --- entropy approximations ---
    stats['H_count'] = spike_count_entropy(flat)
    stats['H_indep'] = independent_entropy(flat)
    
    # kNN on subsample to reduce cost
    sample_size = max(1000, int(T*sample_frac))
    idx = np.random.choice(T, sample_size, replace=False)
    stats['H_knn'] = knn_entropy(flat[idx,:])

    # --- compression ---
    raw_sparse, zlib_sparse = encode_sparse_ids(flat)
    stats['sparse_bytes'] = raw_sparse
    stats['sparse_zlib_bytes'] = zlib_sparse

    raw_run, zlib_run = encode_run_length(flat)
    stats['run_bytes'] = raw_run
    stats['run_zlib_bytes'] = zlib_run

    return stats




# ---- main function ----
def evaluate_partition_entropy_reduced_statistics(
    deployment_configuration: List[List[int]],
    spikes_record: np.ndarray,
    compute_entropy: bool = True,
    pattern_max_K: int = 512,
    pattern_topk: int = 5,
    base: int = 2
):
    """
    Compute many reduced statistics for each partition.

    Parameters
    ----------
    deployment_configuration : List[List[int]]
        i-th element: list of population IDs assigned to partition i.
    spikes_record : np.ndarray
        Shape (T, n_populations, n_neurons_per_pop), binary {0,1}.
    compute_entropy : bool
        Whether to compute entropy-related metrics (can be expensive for large K).
    pattern_max_K : int
        Safety: only compute empirical pattern entropy if K <= pattern_max_K.
    pattern_topk : int
        Return top-k patterns and their frequencies (for small K).
    base : int
        Log base for entropy (2 for bits).

    Returns
    -------
    Dict[partition_idx, stats_dict]
    """
    T, n_pop, n_neurons = spikes_record.shape
    results: Dict[int, Dict[str, object]] = {}

    for pidx, pops in enumerate(deployment_configuration):
        part = spikes_record[:, pops, :]       # (T, P, Np)
        T_, P, Np = part.shape
        assert T_ == T
        K = P * Np
        flat = part.reshape(T, K)              # (T, K)
        counts = flat.sum(axis=1).astype(float)  # counts per time (T,)
        frac = _safe_div(counts, K)              # fraction firing per time
        p_i = flat.mean(axis=0)                  # per-neuron firing prob over time

        stats: Dict[str, object] = {}
        stats["T"] = int(T)
        stats["K"] = int(K)

        # ---- basic count statistics ----
        stats["count.mean"] = float(counts.mean())
        stats["count.std"]  = float(counts.std(ddof=0))
        stats["count.min"]  = float(counts.min()) if counts.size > 0 else 0.0
        stats["count.max"]  = float(counts.max()) if counts.size > 0 else 0.0
        stats["count.median"] = float(np.median(counts))
        stats["count.percentiles"] = {
            "p10": float(np.percentile(counts, 10)),
            "p25": float(np.percentile(counts, 25)),
            "p75": float(np.percentile(counts, 75)),
            "p90": float(np.percentile(counts, 90)),
        }
        stats["count.skewness"] = _skewness(counts)
        stats["count.excess_kurtosis"] = _excess_kurtosis(counts)

        # ---- dynamical statistics ----
        stats["count.CV"] = float(stats["count.std"] / (stats["count.mean"] + 1e-12))
        stats["count.Fano"] = float(np.var(counts, ddof=0) / (stats["count.mean"] + 1e-12))
        stats["count.lag1_autocorr"] = _lag1_autocorr(counts)
        stats["count.burstiness"] = _burstiness(counts)

        # ---- sparsity / occupancy ----
        stats["fraction.mean"] = float(frac.mean())
        stats["fraction.std"]  = float(frac.std(ddof=0))
        stats["fraction.silent_fraction"] = float((counts == 0).sum() / max(1, counts.size))
        stats["fraction.full_active_fraction"] = float((counts == K).sum() / max(1, counts.size))
        stats["neuron_mean_firing_rate"] = float(p_i.mean())   # per neuron average firing prob

        # ---- pattern statistics (small K) ----
        if K <= pattern_max_K:
            # pattern keys
            keys = [bytes(row.astype(np.uint8)) for row in flat]
            cnts = Counter(keys)
            total = sum(cnts.values())
            uniq = len(cnts)
            stats["pattern.unique_count"] = int(uniq)
            # top-k pattern freqs
            top = cnts.most_common(pattern_topk)
            stats["pattern.topk"] = [(k, v / total) for (k, v) in top]
            # empirical pattern entropy
            if compute_entropy:
                probs = np.array(list(cnts.values()), dtype=float) / total
                stats["pattern.empirical_entropy"] = float(_entropy_from_probs(probs, base=base))
            else:
                stats["pattern.empirical_entropy"] = float("nan")
        else:
            # large K approximation
            stats["pattern.unique_count"] = float("nan")
            stats["pattern.topk"] = []

            if compute_entropy:
                # --- Lower bound: empirical observed patterns ---
                keys = [bytes(row.astype(np.uint8)) for row in flat]
                cnts = Counter(keys)
                probs_obs = np.array(list(cnts.values()), dtype=float) / len(keys)
                H_obs = _entropy_from_probs(probs_obs, base=base)
                H_obs_mm = _miller_madow_correction(H_obs, m=len(cnts), T=len(keys))
                
                # --- Upper bound: assume neurons independent ---
                freqs = flat.mean(axis=0)
                H_indep = np.sum(_binary_entropy(freqs, base=base))

                # --- Count-based proxy ---
                counts = flat.sum(axis=1)
                probs_cnt = np.bincount(counts, minlength=K+1) / len(counts)
                H_count = _entropy_from_probs(probs_cnt, base=base)

                # --- Optional: kNN-based approximate configuration entropy ---
                try:
                    H_knn = _knn_entropy_estimator(flat, k=3, base=base)
                except Exception:
                    H_knn = float("nan")

                stats["pattern.empirical_entropy_lower"] = float(H_obs_mm)
                stats["pattern.empirical_entropy_upper"] = float(H_indep)
                stats["pattern.empirical_entropy_countproxy"] = float(H_count)
                stats["pattern.empirical_entropy_knn"] = float(H_knn)
            else:
                stats["pattern.empirical_entropy_lower"] = float("nan")
                stats["pattern.empirical_entropy_upper"] = float("nan")
                stats["pattern.empirical_entropy_countproxy"] = float("nan")
                stats["pattern.empirical_entropy_knn"] = float("nan")

        # ---- entropy-related metrics (count & fraction) ----
        if compute_entropy:
            # count empirical H(N)
            hist = np.bincount(counts.astype(np.int64), minlength=K+1).astype(float)
            probs_count = hist / hist.sum() if hist.sum() > 0 else hist
            stats["count_empirical_entropy"] = float(_entropy_from_probs(probs_count, base=base))

            # binomial exact per time (if K moderate)
            if K > 0:
                # use exact calculation if K not huge (warn user if K large)
                if K <= 2000:  # heuristic threshold
                    Ht = _binomial_entropy_exact(K, frac, base=base)
                    stats["count_binom_exact_mean"] = float(Ht.mean())
                    stats["count_binom_exact_std"] = float(Ht.std(ddof=0))
                else:
                    # fallback: upper bound K * h(p_t)
                    Ht = K * _binary_entropy(frac, base=base)
                    stats["count_binom_exact_mean"] = float(Ht.mean())
                    stats["count_binom_exact_std"] = float(Ht.std(ddof=0))

            # fast upper bound
            Ht_ub = K * _binary_entropy(frac, base=base)
            stats["count_binom_upper_mean"] = float(Ht_ub.mean())
            stats["count_binom_upper_std"]  = float(Ht_ub.std(ddof=0))

            # fraction binary entropy
            Hf = _binary_entropy(frac, base=base)
            stats["fraction_binary_entropy_mean"] = float(Hf.mean())
            stats["fraction_binary_entropy_std"]  = float(Hf.std(ddof=0))

            # pattern independence upper bound (sum of neuron-wise entropies)
            stats["pattern_independence_entropy"] = float(_binary_entropy(p_i, base=base).sum())

        else:
            stats["count_empirical_entropy"] = float("nan")
            stats["count_binom_exact_mean"] = float("nan")
            stats["count_binom_exact_std"] = float("nan")
            stats["count_binom_upper_mean"] = float("nan")
            stats["count_binom_upper_std"] = float("nan")
            stats["fraction_binary_entropy_mean"] = float("nan")
            stats["fraction_binary_entropy_std"] = float("nan")
            stats["pattern_independence_entropy"] = float("nan")

        # --- double-peak spike count statistics ---
        counts = flat.sum(axis=1)
        hist_counts = np.bincount(counts, minlength=K+1)
        prob_counts = hist_counts / hist_counts.sum()

        # simple double-peak score
        double_peak_score = prob_counts[0] + prob_counts[-1]
        stats["spike_count_double_peak_score"] = double_peak_score
        stats["spike_count_double_peak_norm"] = double_peak_score / 2.0
        stats["spike_count_peak_probs"] = [prob_counts[0], prob_counts[-1]]

        results[pidx] = stats

    return results

def evaluate_compressions(deployment_configuration: List[List[int]], spikes_record: np.ndarray, K = 512):
    T, n_pops, n_neurons_per_pop = spikes_record.shape
    max_population_one_core = max([len(neuron_population_ids) for neuron_population_ids in deployment_configuration])
    n_partitions = len(deployment_configuration)  # assuming each population is a partition
    padded_spikes = np.zeros((T, n_partitions, max_population_one_core * n_neurons_per_pop))
    for p in range(n_partitions):
        pop_size = len(deployment_configuration[p])
        # enumeration id and population id in deployment_configuration
        for pop_index, pop in enumerate(deployment_configuration[p]):
            # place to p-th partition's record.
            padded_spikes[:, p, pop_index * n_neurons_per_pop: pop_index * n_neurons_per_pop +  n_neurons_per_pop] = spikes_record[:, pop, :].reshape(T, -1)

    final_spikes = padded_spikes.reshape(T * n_partitions, max_population_one_core * n_neurons_per_pop)

    return evaluate_partition_compression_large_K(final_spikes, K=max_population_one_core * 256)
    
class VirtualNeuromorphicHardware():
    def __init__(self, configuration: HardwareConfiguration = None):
        self.configuration = configuration
        self.performance_monitor : PerformanceMonitor = PerformanceMonitor()
        
        
    def set_configuration(self, configuration: HardwareConfiguration):
        self.configuration = configuration
                
    def do_simulation(self, n_simulation_times, dt, deployment_configuration, network: Network, I_ext):
        if self.configuration is None:
            return None
        
        hardware_configuration = self.configuration
        performance_monitor = self.performance_monitor
        
        time = np.arange(0, n_simulation_times, dt)


        V_trace = np.zeros((len(time), network.num_pops, network.neurons_per_pop))
        spikes_record = np.zeros((len(time), network.num_pops, network.neurons_per_pop), dtype=bool)

        performance_monitor.initialize(hardware_configuration)
        

        # Initialize spike state for previous step (per population)
        S_prev = np.zeros((network.num_pops, network.neurons_per_pop))
        performance_monitor.step_simulation_initialization_cost(hardware_configuration=hardware_configuration)

        for t_i, t in tqdm(enumerate(time)):
            # Compute inputs to each population from all populations
            inputs = np.zeros((network.num_pops, network.neurons_per_pop))
            for target_pop in range(network.num_pops):
                # Sum over all source populations
                for source_pop in range(network.num_pops):
                    inputs[target_pop] += network.connection_matrices[target_pop, source_pop] @ S_prev[source_pop]
                    performance_monitor.step_population_input_intergation_cost(hardware_configuration, deployment_configuration, source_pop, target_pop, S_prev, t_i, t)
                    
            # Add external input
            total_input = inputs + I_ext[t_i]
            performance_monitor.step_process_external_input_cost(hardware_configuration, deployment_configuration, inputs, I_ext, t_i, t)

            # Step populations independently
            for p in range(network.num_pops):
                spikes = network.populations[p].step(total_input[p], t)
                V_trace[t_i, p] = network.populations[p].V
                spikes_record[t_i, p] = spikes
                S_prev[p] = spikes
                performance_monitor.step_spiking_cost(hardware_configuration, deployment_configuration, spikes, t_i, t)

        spike_times_all = []
        neuron_indices_all = []
        performance_monitor.evaluated_entropy = evaluate_partition_entropy_reduced_statistics(deployment_configuration, spikes_record)
        performance_monitor.evaluated_compressions = evaluate_compressions(deployment_configuration=deployment_configuration, spikes_record=spikes_record)
        assert performance_monitor.evaluated_entropy is not None
        assert performance_monitor.evaluated_compressions is not None

        for pop_idx in range(network.num_pops):
            for neuron_idx in range(network.neurons_per_pop):
                # Get spike times for this neuron
                stimes = network.populations[pop_idx].spike_times[neuron_idx]
                # Global neuron ID (pop * neurons_per_pop + neuron)
                global_id = pop_idx * network.neurons_per_pop + neuron_idx
                spike_times_all.extend(stimes)
                neuron_indices_all.extend([global_id] * len(stimes))

        spike_times_all = np.array(spike_times_all)
        neuron_indices_all = np.array(neuron_indices_all)
        return spike_times_all, neuron_indices_all