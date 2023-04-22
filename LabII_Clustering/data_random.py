from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import cv2

class BlobPlotter:
    def __init__(self, n_samples=500, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True, random_state=1):
        self.n_samples = n_samples
        self.n_features = n_features
        self.centers = centers
        self.cluster_std = cluster_std
        self.center_box = center_box
        self.shuffle = shuffle
        self.random_state = random_state
    
    def generate_data(self):
        self.X, self.y = make_blobs(n_samples=self.n_samples, n_features=self.n_features, centers=self.centers, cluster_std=self.cluster_std, center_box=self.center_box, shuffle=self.shuffle, random_state=self.random_state)
    
    def plot_data(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='viridis')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Generated Dataset')
        plt.savefig(".\plots\plot_org.jpeg")
        img=cv2.imread(".\plots\plot_org.jpeg")
        res, im_png = cv2.imencode(".jpeg", img)
        return im_png


