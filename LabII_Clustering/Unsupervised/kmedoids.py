import numpy as np

class KMedoids:
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
                new_centroids[j] = np.median(X[self.labels == j], axis=0)
                
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
    

class KMedoids_2:
    def __init__(self, n_clusters=2, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
    
    def fit(self, X):
        # Randomly initialize medoids
        #rng = np.random.RandomState(self.random_state)
        rng = np.random.RandomState(2)
        self.medoids = rng.choice(X.shape[0], self.n_clusters, replace=False)
        
        for i in range(self.max_iter):
            # Assign each point to the nearest medoid.
            distances = np.abs(X[:, np.newaxis] - X[self.medoids])
            cluster_labels = np.argmin(np.sum(distances, axis=2), axis=1)
            
            # Updating the medoids
            for j in range(self.n_clusters):
                mask = cluster_labels == j
                cluster_points = X[mask]
                cluster_distances = np.sum(np.abs(cluster_points[:, np.newaxis] - cluster_points), axis=2)
                costs = np.sum(cluster_distances, axis=1)
                best_medoid_idx = np.argmin(costs)
                self.medoids[j] = np.where(mask)[0][best_medoid_idx]
        
        self.cluster_labels_ = cluster_labels
        self.medoids_ = [X[i,:] for i in self.medoids]
        return self.medoids_, self.cluster_labels_
    
    def predict(self, X):
        distances = np.abs(X[:, np.newaxis] - X[self.medoids])
        cluster_labels = np.argmin(np.sum(distances, axis=2), axis=1)
        return cluster_labels