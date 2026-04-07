"""
fuzzy_cmeans.py
===============
Self-contained Fuzzy C-Means implementation with full NaN/missing-value support.
No external clustering libraries required — only NumPy, Matplotlib, scikit-learn
(for dataset generation / evaluation only).

Usage
-----
    python fuzzy_cmeans.py                  # runs built-in demo
    python fuzzy_cmeans.py --n_clusters 3   # custom number of clusters

Install dependencies
--------------------
    pip install numpy matplotlib scikit-learn

NaN handling
------------
Pass nan_policy to fuzzy_cmeans():
    "error"   – raise if any NaN found (default, safe)
    "drop"    – silently drop rows that contain any NaN
    "impute"  – mean-impute each feature before running FCM
    "partial" – NaN-aware partial distances: missing features are skipped
                per-pair; works best when missingness is sparse/random
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


# ════════════════════════════════════════════════════════════════════════════
# NaN pre-processing helpers
# ════════════════════════════════════════════════════════════════════════════

def summarise_nans(X: np.ndarray) -> dict:
    """
    Return a summary of missing values in X.

    Returns
    -------
    dict with keys:
        total_missing   int   – total NaN count
        missing_pct     float – % of all values that are NaN
        rows_with_nan   int   – number of rows containing at least one NaN
        col_missing_pct array – per-column NaN percentage
    """
    mask = np.isnan(X)
    return {
        "total_missing":  int(mask.sum()),
        "missing_pct":    float(mask.mean() * 100),
        "rows_with_nan":  int(mask.any(axis=1).sum()),
        "col_missing_pct": mask.mean(axis=0) * 100,
    }


def impute_mean(X: np.ndarray) -> np.ndarray:
    """Replace each NaN with the column (feature) mean. Returns a copy."""
    X = X.copy().astype(float)
    col_means = np.nanmean(X, axis=0)          # NaN-safe column means
    nan_mask = np.isnan(X)
    col_idx = np.where(nan_mask.any(axis=0))[0]
    for j in col_idx:
        X[nan_mask[:, j], j] = col_means[j]
    return X


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
    Weighted cluster centres — NaN-safe via np.nansum.

    centre_j[d] = Σ_i (u_ij^m · x_id) / Σ_i u_ij^m   (NaN x_id excluded)
    """
    Um = U ** m                                        # (n, c)
    # numerator: (c, d) — np.nansum ignores NaN contributions
    numer = np.nansum(Um[:, :, None] * X[:, None, :], axis=0)   # (c, d)
    # denominator: only rows where feature d is observed
    obs_mask = (~np.isnan(X)).astype(float)            # (n, d): 1 if observed
    denom = (Um[:, :, None] * obs_mask[:, None, :]).sum(axis=0)  # (c, d)
    denom = np.fmax(denom, 1e-10)
    return numer / denom                               # (c, d)


def _partial_distance(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    NaN-aware Euclidean distance: missing features are skipped, and the
    squared sum is rescaled by d / n_observed so units stay comparable.

    Returns dist of shape (n, c).
    """
    n, d = X.shape
    c = centers.shape[0]
    diff = X[:, None, :] - centers[None, :, :]        # (n, c, d)
    obs  = (~np.isnan(diff)).astype(float)             # (n, c, d) — 1 if observed
    sq   = np.where(np.isnan(diff), 0.0, diff ** 2)   # (n, c, d) — NaN → 0
    n_obs = obs.sum(axis=2)                            # (n, c) — features used
    n_obs = np.fmax(n_obs, 1.0)
    dist = np.sqrt(sq.sum(axis=2) * d / n_obs)        # rescale to full-d units
    return dist


def _compute_membership(
    X: np.ndarray, centers: np.ndarray, m: float, partial: bool = False
) -> np.ndarray:
    """
    Update membership matrix from distances.

    u_ij = 1 / Σ_k (d_ij / d_ik)^(2/(m-1))
    """
    if partial:
        dist = _partial_distance(X, centers)
    else:
        diff = X[:, None, :] - centers[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)

    dist = np.fmax(dist, 1e-10)
    exp = 2.0 / (m - 1)
    ratio = (dist[:, :, None] / dist[:, None, :]) ** exp
    return 1.0 / ratio.sum(axis=2)


def fuzzy_cmeans(
    X: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    max_iter: int = 150,
    tol: float = 1e-4,
    seed: int = 42,
    nan_policy: str = "error",
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
    nan_policy : how to handle NaN / missing values
                 "error"   – raise ValueError if any NaN present  (default)
                 "drop"    – remove rows with any NaN before clustering
                 "impute"  – replace each NaN with its column mean
                 "partial" – NaN-aware partial distances (best for sparse missingness)

    Returns
    -------
    dict with keys:
        centers      (c, d)  – cluster centres
        U            (n, c)  – membership matrix (rows sum to 1)
        labels       (n,)    – hard labels via argmax(U)
        history      list    – objective value at each iteration
        n_iter       int     – actual iterations run
        fpc          float   – fuzzy partition coefficient (1 = crisp, 1/c = fully fuzzy)
        dropped_idx  array   – row indices dropped (only when nan_policy="drop")
        nan_summary  dict    – output of summarise_nans() on the original X
    """
    if m <= 1:
        raise ValueError("Fuzziness exponent m must be > 1.")

    X = np.array(X, dtype=float)
    has_nan = np.isnan(X).any()
    nan_summary = summarise_nans(X) if has_nan else None
    dropped_idx = np.array([], dtype=int)

    # ── NaN pre-processing ──────────────────────────────────────────────────
    if has_nan:
        if nan_policy == "error":
            info = summarise_nans(X)
            raise ValueError(
                f"X contains {info['total_missing']} NaN values "
                f"({info['missing_pct']:.1f}% of data, "
                f"{info['rows_with_nan']} rows affected). "
                "Set nan_policy='drop', 'impute', or 'partial'."
            )
        elif nan_policy == "drop":
            nan_rows = np.isnan(X).any(axis=1)
            dropped_idx = np.where(nan_rows)[0]
            X = X[~nan_rows]
            print(f"[nan_policy='drop'] Removed {len(dropped_idx)} rows. "
                  f"{len(X)} rows remain.")
        elif nan_policy == "impute":
            n_missing = int(np.isnan(X).sum())
            X = impute_mean(X)
            print(f"[nan_policy='impute'] Imputed {n_missing} missing values with column means.")
        elif nan_policy == "partial":
            print(f"[nan_policy='partial'] Using partial (NaN-aware) distances.")
        else:
            raise ValueError(f"Unknown nan_policy='{nan_policy}'. "
                             "Choose from: 'error', 'drop', 'impute', 'partial'.")

    partial = (nan_policy == "partial") and has_nan

    # ── Main FCM loop ───────────────────────────────────────────────────────
    n_samples = X.shape[0]
    U = _init_membership(n_samples, n_clusters, seed)
    history = []

    for iteration in range(max_iter):
        U_old = U.copy()
        centers = _compute_centers(X, U, m)
        U = _compute_membership(X, centers, m, partial=partial)

        # Objective (weighted sum of squared distances, NaN-safe)
        if partial:
            dist_sq = _partial_distance(X, centers) ** 2
        else:
            diff = X[:, None, :] - centers[None, :, :]
            dist_sq = np.linalg.norm(diff, axis=-1) ** 2
        objective = float(((U ** m) * dist_sq).sum())
        history.append(objective)

        if np.max(np.abs(U - U_old)) < tol:
            break

    labels = np.argmax(U, axis=1)
    fpc = float((U ** 2).sum() / n_samples)

    return {
        "centers":     centers,
        "U":           U,
        "labels":      labels,
        "history":     history,
        "n_iter":      iteration + 1,
        "fpc":         fpc,
        "dropped_idx": dropped_idx,
        "nan_summary": nan_summary,
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

    # ── Inject ~10% missing values for the demo ─────────────────────────────
    rng = np.random.default_rng(args.seed)
    nan_mask = rng.random(X.shape) < 0.10
    X_missing = X.copy()
    X_missing[nan_mask] = np.nan

    info = summarise_nans(X_missing)
    print(f"\nInjected NaNs: {info['total_missing']} total  "
          f"({info['missing_pct']:.1f}%)  |  "
          f"{info['rows_with_nan']} rows affected")
    print(f"Per-column NaN %: {info['col_missing_pct'].round(1)}")

    print(f"\nRunning FCM  (c={args.n_clusters}, m={args.m}, nan_policy='partial') …")
    result = fuzzy_cmeans(
        X_missing,
        n_clusters=args.n_clusters,
        m=args.m,
        seed=args.seed,
        nan_policy="partial",
    )

    print(f"\nConverged in {result['n_iter']} iterations")
    print(f"Fuzzy Partition Coefficient (FPC): {result['fpc']:.4f}")
    # silhouette needs complete rows — use imputed version for scoring only
    X_imp = impute_mean(X_missing)
    sil = silhouette_score(X_imp, result["labels"])
    print(f"Silhouette Score (hard labels):    {sil:.4f}")
    print("\nCluster centres:")
    for i, c in enumerate(result["centers"]):
        print(f"  C{i}: {c.round(3)}")

    if not args.no_plot:
        fig = plot_results(
            X_imp, result,
            title=f"FCM  c={args.n_clusters}, m={args.m}  [nan_policy='partial']",
        )
        plt.show()


if __name__ == "__main__":
    main()
