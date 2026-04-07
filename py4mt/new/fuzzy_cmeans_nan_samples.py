"""
fuzzy_cmeans.py
===============
Self-contained Fuzzy C-Means with NaN support and high-dimensional (n << p) handling.
No external clustering libraries required — only NumPy, Matplotlib, scikit-learn.

Usage
-----
    python fuzzy_cmeans.py                        # standard demo (n >> p)
    python fuzzy_cmeans.py --hd                   # high-dim demo  (n << p)
    python fuzzy_cmeans.py --n_clusters 3 --hd    # custom clusters, high-dim mode

Install dependencies
--------------------
    pip install numpy matplotlib scikit-learn

NaN handling  (nan_policy parameter)
-------------------------------------
    "error"   – raise if any NaN found (default, safe)
    "drop"    – remove rows that contain any NaN
    "impute"  – mean-impute each feature before running FCM
    "partial" – NaN-aware partial distances (best for sparse/random missingness)

High-dimensional handling  (highdim_policy parameter)
------------------------------------------------------
    "auto"    – run distance_concentration_check(); apply fix automatically (default)
    "none"    – skip all checks and preprocessing (use when you know n >> p)
    "pca"     – always reduce with PCA before clustering
    "cosine"  – always use cosine distance instead of Euclidean
    "warn"    – check and warn but do not modify X or distance metric
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist


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
# High-dimensional diagnostics & preprocessing
# ════════════════════════════════════════════════════════════════════════════

def distance_concentration_check(X: np.ndarray, sample: int = 500) -> dict:
    """
    Measure how concentrated pairwise distances are.

    In high dimensions all distances converge to the same value, making
    the membership ratio d_ij/d_ik → 1 and every membership → 1/c.

    Uses the Coefficient of Variation (CV = std/mean) of pairwise distances.

    Parameters
    ----------
    X      : (n, p) array  — will be subsampled if n > `sample`
    sample : max rows to use (pdist is O(n²))

    Returns
    -------
    dict with keys:
        cv          float  – coefficient of variation of pairwise distances
        severity    str    – "ok" | "moderate" | "severe" | "critical"
        recommendation str – plain-English suggestion
        n_used      int    – rows actually used
        mean_dist   float
        std_dist    float
    """
    rng = np.random.default_rng(0)
    n = X.shape[0]
    idx = rng.choice(n, min(n, sample), replace=False)
    X_s = X[idx]

    # drop NaN rows for this diagnostic
    X_s = X_s[~np.isnan(X_s).any(axis=1)]
    if len(X_s) < 4:
        return {"cv": np.nan, "severity": "unknown",
                "recommendation": "Too few complete rows to diagnose.", "n_used": len(X_s),
                "mean_dist": np.nan, "std_dist": np.nan}

    dists = pdist(X_s, metric="euclidean")
    mean_d, std_d = float(dists.mean()), float(dists.std())
    cv = std_d / mean_d if mean_d > 0 else 0.0

    if cv >= 0.5:
        severity = "ok"
        rec = "Distances are well-spread. Standard FCM should work fine."
    elif cv >= 0.2:
        severity = "moderate"
        rec = ("Mild concentration. Consider StandardScaler + PCA, or set "
               "highdim_policy='pca' / 'cosine'.")
    elif cv >= 0.05:
        severity = "severe"
        rec = ("Strong concentration. Dimensionality reduction (PCA/UMAP) "
               "or cosine distance is strongly recommended.")
    else:
        severity = "critical"
        rec = ("All distances are nearly identical — FCM will produce "
               "meaningless uniform memberships. You must reduce dimensions first.")

    return {"cv": cv, "severity": severity, "recommendation": rec,
            "n_used": len(X_s), "mean_dist": mean_d, "std_dist": std_d}


def pca_reduce(
    X: np.ndarray,
    variance_threshold: float = 0.95,
    max_components: int | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, PCA, StandardScaler]:
    """
    Standardise then PCA-reduce X to retain `variance_threshold` of variance.

    Parameters
    ----------
    X                  : (n, p)
    variance_threshold : cumulative explained variance to retain (default 0.95)
    max_components     : hard cap on components (None = no cap)
    scaler             : pass a pre-fit StandardScaler to reuse (e.g. at predict time)

    Returns
    -------
    X_reduced : (n, k)
    pca       : fitted PCA object
    scaler    : fitted StandardScaler object
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    n_comp = min(X.shape[0], X.shape[1])   # max feasible components
    if max_components:
        n_comp = min(n_comp, max_components)

    pca = PCA(n_components=n_comp, svd_solver="full")
    pca.fit(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance_threshold) + 1)
    k = max(k, 1)

    pca_k = PCA(n_components=k, svd_solver="full")
    X_reduced = pca_k.fit_transform(X_scaled)

    print(f"[pca_reduce] {X.shape[1]}→{k} components  "
          f"({cumvar[k-1]*100:.1f}% variance retained)")
    return X_reduced, pca_k, scaler


def _cosine_distance(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Cosine distance ∈ [0, 2]:  1 − cos(x, c).
    Much less affected by dimensionality than Euclidean.

    Returns (n, c) distance matrix.
    """
    X_norm = X / np.fmax(np.linalg.norm(X, axis=1, keepdims=True), 1e-10)
    C_norm = centers / np.fmax(np.linalg.norm(centers, axis=1, keepdims=True), 1e-10)
    return 1.0 - (X_norm @ C_norm.T)


def suggest_max_clusters(n_samples: int) -> int:
    """Conservative upper bound: floor(sqrt(n))."""
    return max(2, int(np.floor(np.sqrt(n_samples))))




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
    X: np.ndarray, centers: np.ndarray, m: float,
    partial: bool = False, cosine: bool = False,
) -> np.ndarray:
    """
    Update membership matrix from distances.

    u_ij = 1 / Σ_k (d_ij / d_ik)^(2/(m-1))
    """
    if cosine:
        dist = _cosine_distance(X, centers)
    elif partial:
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
    highdim_policy: str = "auto",
    pca_variance: float = 0.95,
) -> dict:
    """
    Fuzzy C-Means clustering.

    Parameters
    ----------
    X               : array of shape (n_samples, n_features)
    n_clusters      : number of clusters  (c)
    m               : fuzziness exponent  (>1). m=2 is the standard default.
                      Smaller → crisper partitions; larger → softer overlaps.
    max_iter        : maximum number of iterations
    tol             : convergence threshold on the max membership change
    seed            : random seed for reproducibility
    nan_policy      : "error" | "drop" | "impute" | "partial"
    highdim_policy  : "auto" | "none" | "pca" | "cosine" | "warn"
                      Controls behaviour when n << p or distances are concentrated.
                      "auto"   – diagnose and apply the best fix automatically
                      "none"   – skip all high-dim logic
                      "pca"    – always PCA-reduce before clustering
                      "cosine" – always use cosine distance
                      "warn"   – diagnose and print warning but do not modify data
    pca_variance    : cumulative variance to retain when highdim_policy uses PCA
                      (default 0.95)

    Returns
    -------
    dict with keys:
        centers         (c, d)   cluster centres (in PCA space if PCA was applied)
        U               (n, c)   membership matrix (rows sum to 1)
        labels          (n,)     hard labels via argmax(U)
        history         list     objective value at each iteration
        n_iter          int      actual iterations run
        fpc             float    fuzzy partition coefficient
        dropped_idx     array    row indices dropped (nan_policy="drop" only)
        nan_summary     dict     NaN diagnostics on original X (or None)
        hd_check        dict     distance concentration diagnostics (or None)
        hd_applied      str      which high-dim strategy was actually used
        pca             object   fitted PCA  (or None)
        scaler          object   fitted StandardScaler (or None)
    """
    if m <= 1:
        raise ValueError("Fuzziness exponent m must be > 1.")

    X = np.array(X, dtype=float)
    n, p = X.shape

    # ── NaN pre-processing ──────────────────────────────────────────────────
    has_nan = np.isnan(X).any()
    nan_summary = summarise_nans(X) if has_nan else None
    dropped_idx = np.array([], dtype=int)

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
            print(f"[nan_policy='impute'] Imputed {n_missing} values with column means.")
        elif nan_policy == "partial":
            print("[nan_policy='partial'] Using partial (NaN-aware) distances.")
        else:
            raise ValueError(f"Unknown nan_policy='{nan_policy}'. "
                             "Choose: 'error', 'drop', 'impute', 'partial'.")

    partial = (nan_policy == "partial") and has_nan

    # ── High-dimensional preprocessing ─────────────────────────────────────
    hd_check = None
    hd_applied = "none"
    pca_obj = None
    scaler_obj = None
    use_cosine = False

    c_max = suggest_max_clusters(len(X))
    if n_clusters > c_max:
        print(f"[highdim] WARNING: n_clusters={n_clusters} > recommended max "
              f"{c_max} (= floor(sqrt({len(X)}))). "
              "Results may be degenerate.")

    if highdim_policy != "none":
        hd_check = distance_concentration_check(X)
        severe = hd_check["severity"] in ("severe", "critical")
        moderate = hd_check["severity"] == "moderate"

        print(f"[highdim] Distance CV={hd_check['cv']:.4f}  "
              f"severity={hd_check['severity']}")
        print(f"[highdim] {hd_check['recommendation']}")

        if highdim_policy == "warn":
            hd_applied = "warn_only"

        elif highdim_policy == "pca" or (highdim_policy == "auto" and (severe or moderate)):
            if not partial:   # PCA needs complete data
                X, pca_obj, scaler_obj = pca_reduce(X, variance_threshold=pca_variance)
                hd_applied = "pca"
            else:
                print("[highdim] PCA skipped — partial NaN distances active. "
                      "Impute first if you want PCA.")
                hd_applied = "partial_only"

        elif highdim_policy == "cosine" or (highdim_policy == "auto" and severe and partial):
            use_cosine = True
            hd_applied = "cosine"
            print("[highdim] Using cosine distance.")

    # ── Main FCM loop ───────────────────────────────────────────────────────
    n_samples = X.shape[0]
    U = _init_membership(n_samples, n_clusters, seed)
    history = []

    for iteration in range(max_iter):
        U_old = U.copy()
        centers = _compute_centers(X, U, m)
        U = _compute_membership(X, centers, m, partial=partial, cosine=use_cosine)

        # Objective
        if use_cosine:
            dist_sq = _cosine_distance(X, centers) ** 2
        elif partial:
            dist_sq = _partial_distance(X, centers) ** 2
        else:
            diff = X[:, None, :] - centers[None, :, :]
            dist_sq = np.linalg.norm(diff, axis=-1) ** 2
        history.append(float(((U ** m) * dist_sq).sum()))

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
        "hd_check":    hd_check,
        "hd_applied":  hd_applied,
        "pca":         pca_obj,
        "scaler":      scaler_obj,
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
    p.add_argument("--hd", action="store_true",
                   help="Run high-dimensional demo (n=40, p=200) instead of standard demo")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.hd:
        # ── High-dimensional demo: n=40, p=200 (n << p) ────────────────────
        print("=" * 60)
        print("HIGH-DIMENSIONAL DEMO  (n=40, p=200)")
        print("=" * 60)
        rng = np.random.default_rng(args.seed)
        n, p, c_true = 40, 200, 3
        # 3 latent clusters embedded in 200-d noise
        centers_true = rng.standard_normal((c_true, p)) * 3
        labels_true  = rng.integers(0, c_true, size=n)
        X_hd = centers_true[labels_true] + rng.standard_normal((n, p))

        print(f"\nShape: {X_hd.shape}   (n={n}, p={p})")
        print(f"Recommended max clusters: {suggest_max_clusters(n)}")

        # ── Raw distance concentration check ───────────────────────────────
        print("\n── Distance concentration (raw space) ──")
        chk = distance_concentration_check(X_hd)
        print(f"  CV={chk['cv']:.4f}  severity={chk['severity']}")
        print(f"  {chk['recommendation']}")

        n_cl = min(args.n_clusters, suggest_max_clusters(n))
        print(f"\n── Running FCM with highdim_policy='auto', c={n_cl} ──")
        result = fuzzy_cmeans(
            X_hd, n_clusters=n_cl, m=args.m, seed=args.seed,
            nan_policy="error", highdim_policy="auto",
        )

        print(f"\nConverged in {result['n_iter']} iterations")
        print(f"FPC         : {result['fpc']:.4f}")
        print(f"HD applied  : {result['hd_applied']}")
        if result["pca"] is not None:
            k = result["pca"].n_components_
            var = result["pca"].explained_variance_ratio_.sum()
            print(f"PCA         : {p} → {k} components  ({var*100:.1f}% variance)")

        # Silhouette in original space (impute not needed here, no NaNs)
        if len(set(result["labels"])) > 1:
            # project back via PCA is not straightforward; score in reduced space
            X_eval = result["scaler"].transform(X_hd) if result["scaler"] else X_hd
            if result["pca"]:
                X_eval = result["pca"].transform(X_eval)
            sil = silhouette_score(X_eval, result["labels"])
            print(f"Silhouette  : {sil:.4f}  (in reduced space)")

        if not args.no_plot and result["pca"] is not None:
            # Project to 2-D for visualisation
            X_2d_scaler = StandardScaler().fit(X_hd)
            X_2d = PCA(n_components=2).fit_transform(X_2d_scaler.transform(X_hd))
            _plot_hd_summary(X_hd, X_2d, result, n_cl, args.m)
            plt.show()

    else:
        # ── Standard demo ──────────────────────────────────────────────────
        print("Generating dataset …")
        X, _ = make_blobs(
            n_samples=args.n_samples, centers=args.n_clusters,
            cluster_std=0.9, random_state=args.seed,
        )

        # Inject ~10% NaNs for demo
        rng = np.random.default_rng(args.seed)
        nan_mask = rng.random(X.shape) < 0.10
        X_missing = X.copy()
        X_missing[nan_mask] = np.nan

        info = summarise_nans(X_missing)
        print(f"Injected NaNs: {info['total_missing']} total "
              f"({info['missing_pct']:.1f}%)  |  {info['rows_with_nan']} rows")

        result = fuzzy_cmeans(
            X_missing, n_clusters=args.n_clusters, m=args.m, seed=args.seed,
            nan_policy="partial", highdim_policy="auto",
        )

        print(f"\nConverged in {result['n_iter']} iterations")
        print(f"FPC: {result['fpc']:.4f}")
        X_imp = impute_mean(X_missing)
        sil = silhouette_score(X_imp, result["labels"])
        print(f"Silhouette: {sil:.4f}")

        if not args.no_plot:
            fig = plot_results(X_imp, result,
                               title=f"FCM  c={args.n_clusters}, m={args.m}  [nan_policy='partial']")
            plt.show()


def _plot_hd_summary(X_raw, X_2d, result, n_cl, m):
    """Compact 3-panel figure for the high-dim demo."""
    labels  = result["labels"]
    history = result["history"]
    hd_chk  = result["hd_check"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"FCM high-dim demo  (n<<p)  c={n_cl}, m={m}, "
                 f"hd={result['hd_applied']}", fontsize=12, fontweight="bold")
    cmap = plt.cm.get_cmap("tab10", n_cl)

    # Panel 1: 2-D PCA projection coloured by cluster
    ax = axes[0]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap=cmap, s=60, alpha=0.8)
    ax.set_title("First 2 PCs (raw X) — cluster labels")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # Panel 2: membership matrix heatmap
    ax = axes[1]
    im = ax.imshow(result["U"].T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("Sample"); ax.set_ylabel("Cluster")
    ax.set_yticks(range(n_cl))
    ax.set_yticklabels([f"C{i}" for i in range(n_cl)])
    plt.colorbar(im, ax=ax, label="Membership")
    ax.set_title("Membership matrix")

    # Panel 3: convergence
    ax = axes[2]
    ax.plot(history, color="steelblue", marker="o", markersize=3, linewidth=1.5)
    if hd_chk:
        ax.set_title(f"Convergence  (distance CV={hd_chk['cv']:.3f}, "
                     f"severity={hd_chk['severity']})")
    else:
        ax.set_title("Convergence")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Objective")

    plt.tight_layout()


if __name__ == "__main__":
    main()
