import sklearn.cluster
import numpy as np

def kmeans_missing(X, n_clusters=3, max_iter=10):
    """Perform K-Means clustering on data with missing values.

    Args:
      X: An [n_samples, n_features] array of data to cluster.
      n_clusters: Number of clusters to form.
      max_iter: Maximum number of EM iterations to perform.

    Returns:
      labels: An [n_samples] vector of integer labels.
      centroids: An [n_clusters, n_features] array of cluster centroids.
      X_hat: Copy of X with the missing values filled in.
    """

    # Initialize missing values to their column means
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)


    for i in range(max_iter):
        if i == 0:

            # do multiple random initializations in parallel
            cls = sklearn.cluster.KMeans(n_clusters)
            # cls = sklearn.cluster.MiniBatchKMeans(n_clusters)

            labels = cls.fit_predict(X_hat)
            centroids = cls.cluster_centers_

            X_hat[missing] = centroids[labels][missing]

            prev_labels = labels
            prev_centroids = cls.cluster_centers_
        else:
            # initialize KMeans with the previous set of centroids. this is much
            # faster and makes it easier to check convergence (since labels
            # won't be permuted on every iteration), but might be more prone to
            # getting stuck in local minima.
            cls = sklearn.cluster.MiniBatchKMeans(n_clusters, init=prev_centroids)
            # cls = sklearn.cluster.MiniBatchKMeans(n_clusters, 4, init=prev_centroids)
            labels = cls.fit_predict(X_hat)
            centroids = cls.cluster_centers_

            X_hat[missing] = centroids[labels][missing]

            prev_labels = labels
            prev_centroids = cls.cluster_centers_


        # when the labels have stopped changing then we have converged
        if i > 0 and np.all(labels == prev_labels):
            break



    return labels, centroids, X_hat
