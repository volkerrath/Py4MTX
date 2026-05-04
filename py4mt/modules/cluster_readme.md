# cluster.py — K-means clustering with missing values

`cluster.py` provides a single utility function for K-means clustering on
datasets that contain NaN or non-finite entries.  The standard
`sklearn.cluster.KMeans` does not handle missing values; this module adds
an EM-style imputation loop around it.

---

## Function

### `kmeans_missing(X, n_clusters=3, max_iter=10)`

Perform K-means clustering on data with missing values via iterative
imputation.

**Algorithm:**

1. Initialise missing entries to their column (feature) means.
2. Fit K-means on the imputed array; assign cluster labels.
3. Replace missing entries with the corresponding cluster-centroid values.
4. Repeat steps 2–3 until labels stabilise or `max_iter` is reached.

Iteration 0 uses `KMeans` with multiple random initialisations (parallel
`init` mode).  Subsequent iterations use `MiniBatchKMeans` seeded with the
previous centroids — cheaper and convergence-safe since labels are not
permuted between iterations.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | ndarray, shape (n_samples, n_features) | — | Data array; NaN and ±inf treated as missing |
| `n_clusters` | int | `3` | Number of clusters |
| `max_iter` | int | `10` | Maximum EM iterations |

**Returns:** `(labels, centroids, X_hat)`

| Return value | Shape | Description |
|---|---|---|
| `labels` | (n_samples,) | Integer cluster assignment per sample |
| `centroids` | (n_clusters, n_features) | Final cluster centroids |
| `X_hat` | (n_samples, n_features) | Copy of X with missing values imputed |

**Dependencies:** `numpy`, `scikit-learn`.

---

## Example

```python
import numpy as np
from cluster import kmeans_missing

# 200 samples, 5 features, ~20% missing
rng = np.random.default_rng(0)
X = rng.standard_normal((200, 5))
X[rng.random(X.shape) < 0.2] = np.nan

labels, centroids, X_hat = kmeans_missing(X, n_clusters=4, max_iter=15)
print(f"Cluster sizes: {np.bincount(labels)}")
print(f"Centroids:\n{centroids}")
```

---

## Notes

- The returned `X_hat` has all missing entries filled with the centroid of
  the assigned cluster.  Original non-missing values are unchanged.
- Convergence is checked on label stability, not centroid distance.
  `max_iter` guards against slow-converging cases.
- For large datasets or many clusters, consider increasing `max_iter` or
  reducing `n_clusters` to ensure stability before declaring convergence.

---

Author: Volker Rath (DIAS)
