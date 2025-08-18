import numpy as np
import networkx as nx
import community as community_louvain
from typing import List
from networkx.algorithms.community import kernighan_lin_bisection

# ---------- Helpers ----------

def ensure_symmetric(J: np.ndarray) -> np.ndarray:
    """Symmetrize and zero the diagonal; preserves signs."""
    S = (J + J.T) / 2.0
    np.fill_diagonal(S, 0.0)
    return S

def intra_score(J: np.ndarray, parts: List[List[int]], use_abs: bool = False) -> float:
    """Sum of within-part pair weights (i<j). J expected symmetric for correct halving."""
    W = ensure_symmetric(np.abs(J) if use_abs else J)
    total = 0.0
    for grp in parts:
        g = np.array(grp, dtype=int)
        if len(g) <= 1:
            continue
        sub = W[np.ix_(g, g)]
        total += (np.sum(sub) - np.sum(np.diag(sub))) / 2.0
    return float(total)

def finalize_groups(order: List[int], K: int) -> List[List[int]]:
    return [order[i:i+K] for i in range(0, len(order), K)]

def to_parts_from_assignment(assign: np.ndarray) -> List[List[int]]:
    parts = {}
    for i, c in enumerate(assign):
        parts.setdefault(int(c), []).append(i)
    return list(parts.values())

# ---------- Algorithm 1: Greedy seed-and-fill ----------

def partition_greedy(J: np.ndarray, K: int, use_abs: bool = False) -> List[List[int]]:
    W = ensure_symmetric(np.abs(J) if use_abs else J)
    n = W.shape[0]
    remaining = set(range(n))
    parts = []
    strength = np.sum(W, axis=1)

    while remaining:
        seed = max(remaining, key=lambda i: strength[i])
        group = [seed]
        remaining.remove(seed)
        while len(group) < K and remaining:
            best = max(remaining, key=lambda j: float(np.sum(W[j, group])))
            group.append(best)
            remaining.remove(best)
        parts.append(group)
    return parts

# ---------- Algorithm 2: Capacity-constrained agglomerative (avg-link) ----------

def partition_agglomerative(J: np.ndarray, K: int, use_abs: bool = False) -> List[List[int]]:
    W = ensure_symmetric(np.abs(J) if use_abs else J)
    n = W.shape[0]
    clusters = [[i] for i in range(n)]

    def avg_link(c1, c2):
        c1 = np.array(c1); c2 = np.array(c2)
        if len(c1) == 0 or len(c2) == 0: return -np.inf
        sub = W[np.ix_(c1, c2)]
        return float(np.mean(sub))

    while True:
        best_score = -np.inf
        best_pair = None
        for i in range(len(clusters)):
            if not clusters[i]: continue
            for j in range(i+1, len(clusters)):
                if not clusters[j]: continue
                if len(clusters[i]) + len(clusters[j]) > K: continue
                s = avg_link(clusters[i], clusters[j])
                if s > best_score:
                    best_score = s
                    best_pair = (i, j)
        if best_pair is None or best_score == -np.inf:
            break
        i, j = best_pair
        clusters[i] = clusters[i] + clusters[j]
        clusters[j] = []
    return [c for c in clusters if c]

# ---------- Algorithm 3: Spectral ordering + bin packing ----------

def partition_spectral(J: np.ndarray, K: int, use_abs: bool = False) -> List[List[int]]:
    W = ensure_symmetric(np.abs(J) if use_abs else J)
    vals, vecs = np.linalg.eigh(W)  # symmetric
    leading = vecs[:, -1]
    order = list(np.argsort(-leading))
    return finalize_groups(order, K)

# ---------- Local refinement: pairwise swaps ----------

def refine_local_swaps(J: np.ndarray, parts: List[List[int]], use_abs: bool = False, max_passes: int = 10) -> List[List[int]]:
    W = ensure_symmetric(np.abs(J) if use_abs else J)
    n = W.shape[0]
    labels = -np.ones(n, dtype=int)
    parts = [list(g) for g in parts]
    for c, grp in enumerate(parts):
        for i in grp: labels[i] = c

    current = intra_score(W, parts, use_abs=False)

    def delta_swap(i, j, ci, cj) -> float:
        if ci == cj: return 0.0
        gi = parts[ci]; gj = parts[cj]
        gi_set = set(gi); gj_set = set(gj)
        gi_set.remove(i); gj_set.remove(j)
        d = 0.0
        if gi_set: d -= np.sum(W[i, list(gi_set)])
        if gj_set: d -= np.sum(W[j, list(gj_set)])
        if gj_set: d += np.sum(W[i, list(gj_set)])
        if gi_set: d += np.sum(W[j, list(gi_set)])
        return float(d)

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for ci in range(len(parts)):
            for cj in range(ci+1, len(parts)):
                for ii in range(len(parts[ci])):
                    for jj in range(len(parts[cj])):
                        i = parts[ci][ii]; j = parts[cj][jj]
                        d = delta_swap(i, j, ci, cj)
                        if d > 1e-12:
                            parts[ci][ii], parts[cj][jj] = j, i
                            labels[i], labels[j] = cj, ci
                            current += d
                            improved = True
                            break
                    if improved: break
                if improved: break
            if improved: break
    return parts

# ---------- Simple k-medoids-like ----------

def partition_simple_kmedoids(J: np.ndarray, K: int, use_abs: bool = False) -> List[List[int]]:
    W = ensure_symmetric(np.abs(J) if use_abs else J)
    n = W.shape[0]
    D = np.max(W) - W
    D = ensure_symmetric(D)

    medoids = [np.argmax(np.sum(W, axis=1))]
    for _ in range(1, max(1, n // K)):
        remaining = [i for i in range(n) if i not in medoids]
        dists = [min(D[i, m] for m in medoids) for i in remaining]
        medoids.append(remaining[np.argmax(dists)])

    labels = np.argmin(D[:, medoids], axis=1)
    parts = {}
    for i, l in enumerate(labels):
        parts.setdefault(l, []).append(i)
    parts = list(parts.values())

    final_parts = []
    for grp in parts:
        final_parts.extend([grp[i:i+K] for i in range(0, len(grp), K)])
    return final_parts

# ---------- Louvain (positive weights only) ----------

def partition_louvain(J: np.ndarray, K: int, use_abs: bool = False) -> List[List[int]]:
    W = ensure_symmetric(np.abs(J) if use_abs else J)
    min_w = np.min(J)
    shift = -min_w + 1e-9 if min_w < 0 else 1e-9
    W = J + shift
    
    n = W.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # Use positive couplings only for modularity
    for i in range(n):
        for j in range(i+1, n):
            w = W[i, j]
            if w > 0:
                G.add_edge(i, j, weight=float(w))

    labels_dict = community_louvain.best_partition(G, weight='weight', random_state=42)
    labels = [labels_dict[i] for i in range(n)]
    parts = to_parts_from_assignment(labels)

    final_parts = []
    for grp in parts:
        final_parts.extend(finalize_groups(grp, K) if len(grp) > K else [grp])
    return final_parts

# ---------- Kernighan–Lin (positive weights only, robust) ----------

def partition_kernighan_lin(J: np.ndarray, K: int, use_abs: bool = False) -> List[List[int]]:
    W = ensure_symmetric(np.abs(J) if use_abs else J)
    # shift all weights to be positive
    min_w = np.min(J)
    shift = -min_w + 1e-9 if min_w < 0 else 1e-9
    W = J + shift
    
    n = W.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            w = W[i, j]
            if w > 0:
                G.add_edge(i, j, weight=float(w))

    parts = [list(range(n))]
    final_parts = []
    while parts:
        grp = parts.pop()
        if len(grp) <= K or len(grp) < 2:
            final_parts.append(grp)
            continue
        subG = G.subgraph(grp)
        try:
            half = len(grp) // 2
            init_A = set(grp[:half]); init_B = set(grp[half:])
            A, B = kernighan_lin_bisection(subG, partition=(init_A, init_B), weight='weight')
        except Exception:
            mid = len(grp) // 2
            A, B = set(grp[:mid]), set(grp[mid:])
        parts.extend([list(A), list(B)])
    return final_parts
