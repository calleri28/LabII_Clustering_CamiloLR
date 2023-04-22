import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iters=1000, tol=1e-5):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for i in range(self.max_iters):
            # Assign each data point to the nearest centroid
            distances = self._calc_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for j in range(self.n_clusters):
                new_centroids[j] = np.mean(X[self.labels == j], axis=0)
                
            # Check for convergence
            if np.sum(np.abs(new_centroids - self.centroids)) < self.tol:
                break
                
            self.centroids = new_centroids
            
    def predict(self, X):
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)
        
    def _calc_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances

    def fit_transform(self, X):
        self.fit(X)
        transformed_data = np.zeros((X.shape[0], self.n_clusters))
        for j in range(self.n_clusters):
            transformed_data[:, j] = (self.labels_ == j).astype(int)
        return transformed_data
    

import numpy as np

class KMeans_2():
    def __init__(self, K, max_iters=100):
        self.K = K
        self.max_iters = max_iters

    def fit(self, X):
         # Initialize random centroids
        self.centroids = X[np.random.choice(X.shape[0], self.K, replace=False)]
        for i in range(self.max_iters):
            # Calculate distance between data and centroids
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            # Assign each point to the nearest cluster
            self.labels = np.argmin(distances, axis=0)
            # Update the centroids
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.K)])
            # Check if the centroids have changed
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return self.centroids, self.labels

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        return labels