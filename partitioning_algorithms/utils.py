
import numpy as np
from typing import List



import networkx as nx
import community as community_louvain
from networkx.algorithms.community import kernighan_lin_bisection

# ---------- Utilities ----------

def intra_score(J: np.ndarray, parts: List[List[int]], use_abs: bool = True):
    """Sum of within-part pair weights (i<j), optionally using absolute values."""
    W = np.abs(J) if use_abs else J
    total = 0.0
    for grp in parts:
        g = np.array(grp, dtype=int)
        if len(g) <= 1: 
            continue
        sub = W[np.ix_(g, g)]
        total += (np.sum(sub) - np.sum(np.diag(sub))) / 2.0
    return float(total)

def finalize_groups(order: List[int], K: int) -> List[List[int]]:
    """Chunk an ordering of nodes into groups of size <= K."""
    return [order[i:i+K] for i in range(0, len(order), K)]

def to_parts_from_assignment(assign: np.ndarray) -> List[List[int]]:
    """Convert an array of cluster labels to list-of-lists, preserving label order."""
    parts = {}
    for i, c in enumerate(assign):
        parts.setdefault(int(c), []).append(i)
    return list(parts.values())

# ---------- Algorithm 1: Greedy seed-and-fill by strongest links ----------

def partition_greedy(J: np.ndarray, K: int, use_abs: bool = True):
    """Build groups by repeatedly picking the most 'connected' unassigned node as a seed,
    then filling the group with the strongest linked remaining nodes."""
    W = np.abs(J) if use_abs else J
    n = J.shape[0]
    remaining = set(range(n))
    parts = []

    # Precompute total connection strength for tie-breaking
    strength = np.sum(W, axis=1)

    while remaining:
        # pick seed with highest total strength among remaining
        seed = max(remaining, key=lambda i: strength[i])
        group = [seed]
        remaining.remove(seed)

        if K > 1:
            # iteratively add the remaining node that has highest sum of weights to current group
            while len(group) < K and remaining:
                best = max(
                    remaining,
                    key=lambda j: float(np.sum(W[j, group]))
                )
                group.append(best)
                remaining.remove(best)

        parts.append(group)

    return parts

# ---------- Algorithm 2: Capacity-constrained agglomerative (average-link) ----------

def partition_agglomerative(J: np.ndarray, K: int, use_abs: bool = True):
    """Start with singletons; repeatedly merge the pair of clusters with highest average inter-weight,
    as long as the merged size <= K. When no valid merge remains, freeze those clusters and continue."""
    W = np.abs(J) if use_abs else J
    n = J.shape[0]
    clusters = [[i] for i in range(n)]
    active = [True] * n  # cluster is still mergeable
    # Precompute pair sums to speed up (simple, O(n^3) worst case for small/medium n)
    def avg_link(c1, c2):
        c1 = np.array(c1); c2 = np.array(c2)
        if len(c1) == 0 or len(c2) == 0: return -np.inf
        sub = W[np.ix_(c1, c2)]
        return float(np.mean(sub))

    # While any merge is possible
    while True:
        best_score = -np.inf
        best_pair = None
        for i in range(len(clusters)):
            if not active[i]: 
                continue
            for j in range(i+1, len(clusters)):
                if not active[j]:
                    continue
                if len(clusters[i]) + len(clusters[j]) > K:
                    continue
                s = avg_link(clusters[i], clusters[j])
                if s > best_score:
                    best_score = s
                    best_pair = (i, j)
        if best_pair is None or best_score == -np.inf:
            # No more merges possible among active clusters.
            # If any active cluster has size < K but can't merge, just mark it inactive.
            # Stop when all inactive.
            any_new_inactive = False
            for i in range(len(clusters)):
                if active[i]:
                    active[i] = False
                    any_new_inactive = True
            if not any_new_inactive:
                break
        else:
            i, j = best_pair
            # merge j into i
            clusters[i] = clusters[i] + clusters[j]
            active[i] = True  # merged cluster remains active
            active[j] = False
            clusters[j] = []   # tombstone

    parts = [c for c in clusters if len(c) > 0]
    return parts

# ---------- Algorithm 3: Spectral ordering + bin packing ----------

def partition_spectral(J: np.ndarray, K: int, use_abs: bool = True):
    W = np.abs(J) if use_abs else J
    # Ensure symmetric
    W = (W + W.T) / 2.0
    # Use leading eigenvector of W
    vals, vecs = np.linalg.eigh(W)  # symmetric -> eigh
    leading = vecs[:, -1]
    order = list(np.argsort(-leading))  # descending
    return finalize_groups(order, K)

# ---------- Local refinement: pairwise swaps to improve score ----------

def refine_local_swaps(J: np.ndarray, parts: List[List[int]], use_abs: bool = True, max_passes: int = 10):
    W = np.abs(J) if use_abs else J
    # Build label array
    n = W.shape[0]
    labels = -np.ones(n, dtype=int)
    for c, grp in enumerate(parts):
        for i in grp:
            labels[i] = c
    parts = [list(g) for g in parts]
    current = intra_score(W, parts, use_abs=False)  # W already abs if needed

    def delta_swap(i, j, ci, cj) -> float:
        """Score change from swapping i in ci with j in cj."""
        if ci == cj: 
            return 0.0
        gi = parts[ci]; gj = parts[cj]
        # Remove i from gi, add j to gi; remove j from gj, add i to gj
        # Delta = change in edges incident to i and j within their clusters
        gi_set = set(gi); gj_set = set(gj)
        gi_set.remove(i); gj_set.remove(j)
        # Contributions for i in gj and j in gi (excluding self)
        d = 0.0
        # i leaves gi
        if gi_set:
            d -= np.sum(W[i, list(gi_set)])
        # j leaves gj
        if gj_set:
            d -= np.sum(W[j, list(gj_set)])
        # i enters gj
        if gj_set:
            d += np.sum(W[i, list(gj_set)])
        # j enters gi
        if gi_set:
            d += np.sum(W[j, list(gi_set)])
        return float(d)

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        # try all cross-cluster pairs (could be expensive; stop at first improving move)
        for ci in range(len(parts)):
            for cj in range(ci+1, len(parts)):
                for ii in range(len(parts[ci])):
                    for jj in range(len(parts[cj])):
                        i = parts[ci][ii]
                        j = parts[cj][jj]
                        d = delta_swap(i, j, ci, cj)
                        if d > 1e-12:  # strictly better
                            # perform swap
                            parts[ci][ii], parts[cj][jj] = j, i
                            labels[i], labels[j] = cj, ci
                            current += d
                            improved = True
                            break
                    if improved: break
                if improved: break
            if improved: break

    return parts

def partition_simple_kmedoids(J: np.ndarray, K: int, use_abs: bool = True):
    """
    Simple k-medoids-like partitioning using only NumPy.
    
    Args:
        J : np.ndarray
            Coupling matrix (square, symmetric).
        K : int
            Maximum number of sites per group.
        use_abs : bool
            If True, use absolute coupling values for clustering.
            
    Returns:
        List[List[int]] : List of groups (each group is a list of node indices).
    """
    W = np.abs(J) if use_abs else J
    n = W.shape[0]
    
    # Convert similarity to distance
    D = np.max(W) - W
    D = (D + D.T) / 2.0  # ensure symmetry

    # Choose initial medoids greedily: pick the most connected node first, then farthest from existing medoids
    medoids = [np.argmax(np.sum(W, axis=1))]
    for _ in range(1, max(1, n // K)):
        remaining = [i for i in range(n) if i not in medoids]
        # Farthest-point heuristic: pick node farthest from closest medoid
        dists = [min(D[i, m] for m in medoids) for i in remaining]
        medoids.append(remaining[np.argmax(dists)])
    
    # Assign each node to nearest medoid
    labels = np.argmin(D[:, medoids], axis=1)
    
    # Convert labels to list-of-lists
    parts = {}
    for i, l in enumerate(labels):
        parts.setdefault(l, []).append(i)
    parts = list(parts.values())

    # Split any group larger than K
    final_parts = []
    for grp in parts:
        if len(grp) > K:
            # chunk into multiple groups of size <= K
            for i in range(0, len(grp), K):
                final_parts.append(grp[i:i+K])
        else:
            final_parts.append(grp)
    
    return final_parts




def partition_louvain(J: np.ndarray, K: int, use_abs: bool = True) -> List[List[int]]:
    """Detect communities via Louvain modularity, then split if needed."""
    W = np.abs(J) if use_abs else J
    n = W.shape[0]
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            if W[i,j] > 0:
                G.add_edge(i, j, weight=W[i,j])
    labels_dict = community_louvain.best_partition(G, weight='weight', random_state=42)
    labels = [labels_dict[i] for i in range(n)]
    parts = to_parts_from_assignment(labels)
    final_parts = []
    for grp in parts:
        if len(grp) > K:
            final_parts.extend(finalize_groups(grp, K))
        else:
            final_parts.append(grp)
    return final_parts


def partition_kernighan_lin(J: np.ndarray, K: int, use_abs: bool = True) -> List[List[int]]:
    """Recursively bisect with Kernighan–Lin until all groups ≤ K."""
    W = np.abs(J) if use_abs else J
    n = W.shape[0]
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            if W[i,j] > 0:
                G.add_edge(i, j, weight=W[i,j])

    parts = [list(range(n))]
    final_parts = []
    while parts:
        grp = parts.pop()
        if len(grp) <= K:
            final_parts.append(grp)
        else:
            subG = G.subgraph(grp)
            A, B = kernighan_lin_bisection(subG, weight='weight')
            parts.extend([list(A), list(B)])
    return final_parts
