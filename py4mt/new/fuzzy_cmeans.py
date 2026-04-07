"""
fuzzy_cmeans.py
===============
Self-contained Fuzzy C-Means implementation.
No external clustering libraries required — only NumPy, Matplotlib, scikit-learn
(for dataset generation / evaluation only).

Usage
-----
    python fuzzy_cmeans.py                  # runs built-in demo
    python fuzzy_cmeans.py --n_clusters 3   # custom number of clusters

Install dependencies
--------------------
    pip install numpy matplotlib scikit-learn
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


# ════════════════════════════════════════════════════════════════════════════
# Core algorithm
# ════════════════════════════════════════════════════════════════════════════

def _init_membership(n_samples: int, n_clusters: int, seed: int = 42) -> np.ndarray:
    """Random membership matrix whose rows sum to 1."""
    rng = np.random.default_rng(seed)
    U = rng.random((n_samples, n_clusters))
    return U / U.sum(axis=1, keepdims=True)


def _compute_centers(X: np.ndarray, U: np.ndarray, m: float) -> np.ndarray:
    """
    Weighted cluster centres.

    centre_j = Σ_i (u_ij^m · x_i) / Σ_i u_ij^m
    """
    Um = U ** m                                    # (n, c)
    return (Um.T @ X) / Um.sum(axis=0)[:, None]   # (c, d)


def _compute_membership(X: np.ndarray, centers: np.ndarray, m: float) -> np.ndarray:
    """
    Update membership matrix from distances.

    u_ij = 1 / Σ_k (d_ij / d_ik)^(2/(m-1))
    """
    diff = X[:, None, :] - centers[None, :, :]     # (n, c, d)
    dist = np.linalg.norm(diff, axis=-1)            # (n, c)
    dist = np.fmax(dist, 1e-10)                     # guard: avoid div-by-zero

    exp = 2.0 / (m - 1)
    ratio = (dist[:, :, None] / dist[:, None, :]) ** exp  # (n, c, c)
    return 1.0 / ratio.sum(axis=2)                        # (n, c)


def fuzzy_cmeans(
    X: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    max_iter: int = 150,
    tol: float = 1e-4,
    seed: int = 42,
) -> dict:
    """
    Fuzzy C-Means clustering.

    Parameters
    ----------
    X          : array of shape (n_samples, n_features)
    n_clusters : number of clusters  (c)
    m          : fuzziness exponent  (>1).  m=2 is the standard default.
                 Smaller → crisper partitions; larger → softer overlaps.
    max_iter   : maximum number of iterations
    tol        : convergence threshold on the max membership change
    seed       : random seed for reproducibility

    Returns
    -------
    dict with keys:
        centers   (c, d)  – cluster centres
        U         (n, c)  – membership matrix (rows sum to 1)
        labels    (n,)    – hard labels via argmax(U)
        history   list    – objective value at each iteration
        n_iter    int     – actual iterations run
        fpc       float   – fuzzy partition coefficient (1 = crisp, 1/c = fully fuzzy)
    """
    if m <= 1:
        raise ValueError("Fuzziness exponent m must be > 1.")

    n_samples = X.shape[0]
    U = _init_membership(n_samples, n_clusters, seed)
    history = []

    for iteration in range(max_iter):
        U_old = U.copy()
        centers = _compute_centers(X, U, m)
        U = _compute_membership(X, centers, m)

        # Objective (weighted sum of squared distances)
        diff = X[:, None, :] - centers[None, :, :]
        dist_sq = np.linalg.norm(diff, axis=-1) ** 2
        objective = float(((U ** m) * dist_sq).sum())
        history.append(objective)

        if np.max(np.abs(U - U_old)) < tol:
            break

    labels = np.argmax(U, axis=1)
    fpc = float((U ** 2).sum() / n_samples)   # fuzzy partition coefficient

    return {
        "centers": centers,
        "U": U,
        "labels": labels,
        "history": history,
        "n_iter": iteration + 1,
        "fpc": fpc,
    }


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def predict_membership(X_new: np.ndarray, centers: np.ndarray, m: float = 2.0) -> np.ndarray:
    """
    Compute membership values for new (unseen) points given fitted centres.

    Parameters
    ----------
    X_new   : (n, d)
    centers : (c, d)  – from fuzzy_cmeans result["centers"]
    m       : same fuzziness exponent used during training

    Returns
    -------
    U_new : (n, c) membership matrix
    """
    return _compute_membership(X_new, centers, m)


def choose_n_clusters(X: np.ndarray, cluster_range=range(2, 9), m: float = 2.0, seed: int = 42):
    """
    Evaluate FCM for a range of cluster counts using the
    Fuzzy Partition Coefficient (FPC) and Silhouette Score.

    Higher FPC → more compact partition.
    Higher Silhouette → better-separated clusters.

    Returns a dict mapping n_clusters → {"fpc": ..., "silhouette": ...}
    """
    results = {}
    for c in cluster_range:
        res = fuzzy_cmeans(X, n_clusters=c, m=m, seed=seed)
        sil = silhouette_score(X, res["labels"]) if len(set(res["labels"])) > 1 else float("nan")
        results[c] = {"fpc": res["fpc"], "silhouette": sil}
        print(f"  c={c:2d}  FPC={res['fpc']:.4f}  Silhouette={sil:.4f}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════

def plot_results(X: np.ndarray, result: dict, title: str = "Fuzzy C-Means") -> plt.Figure:
    """Four-panel diagnostic figure."""
    centers = result["centers"]
    U = result["U"]
    labels = result["labels"]
    history = result["history"]
    n_clusters = centers.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    cmap = plt.cm.get_cmap("tab10", n_clusters)

    # ── Panel 1: Hard labels ────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap, alpha=0.5, s=20)
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=220,
               c=range(n_clusters), cmap=cmap, edgecolors="black", linewidths=0.8, zorder=5)
    ax.set_title("Hard Labels (argmax membership)")

    # ── Panel 2: Uncertainty map (entropy of memberships) ──────────────────
    ax = axes[0, 1]
    entropy = -np.sum(U * np.log(U + 1e-10), axis=1)
    sc = ax.scatter(X[:, 0], X[:, 1], c=entropy, cmap="plasma", s=20)
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=220,
               c="white", edgecolors="black", linewidths=0.8, zorder=5)
    plt.colorbar(sc, ax=ax, label="Entropy (high = uncertain)")
    ax.set_title("Membership Entropy")

    # ── Panel 3: Membership heatmap (first 60 points) ──────────────────────
    ax = axes[1, 0]
    n_show = min(60, len(X))
    im = ax.imshow(U[:n_show].T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("Sample index (first 60)")
    ax.set_ylabel("Cluster")
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"C{i}" for i in range(n_clusters)])
    plt.colorbar(im, ax=ax, label="Membership degree")
    ax.set_title("Membership Heatmap")

    # ── Panel 4: Convergence ────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(history, color="steelblue", linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective (weighted SSE)")
    ax.set_title(f"Convergence  ({result['n_iter']} iters, FPC={result['fpc']:.4f})")

    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Demo / CLI entry-point
# ════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="Fuzzy C-Means demo")
    p.add_argument("--n_clusters", type=int, default=4, help="Number of clusters (default 4)")
    p.add_argument("--m", type=float, default=2.0, help="Fuzziness exponent (default 2.0)")
    p.add_argument("--n_samples", type=int, default=300, help="Dataset size (default 300)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
    p.add_argument("--no-plot", action="store_true", help="Skip the plot window")
    return p.parse_args()


def main():
    args = _parse_args()

    print("Generating dataset …")
    X, _ = make_blobs(
        n_samples=args.n_samples,
        centers=args.n_clusters,
        cluster_std=0.9,
        random_state=args.seed,
    )

    print(f"Running FCM  (c={args.n_clusters}, m={args.m}) …")
    result = fuzzy_cmeans(X, n_clusters=args.n_clusters, m=args.m, seed=args.seed)

    print(f"\nConverged in {result['n_iter']} iterations")
    print(f"Fuzzy Partition Coefficient (FPC): {result['fpc']:.4f}")
    sil = silhouette_score(X, result["labels"])
    print(f"Silhouette Score (hard labels):    {sil:.4f}")
    print("\nCluster centres:")
    for i, c in enumerate(result["centers"]):
        print(f"  C{i}: {c.round(3)}")

    if not args.no_plot:
        fig = plot_results(X, result, title=f"FCM  c={args.n_clusters}, m={args.m}")
        plt.show()


if __name__ == "__main__":
    main()
