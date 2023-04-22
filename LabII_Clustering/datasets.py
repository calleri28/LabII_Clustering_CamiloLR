import numpy as np
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import cv2

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.cluster import KMeans


from sklearn.cluster import KMeans, SpectralClustering
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import DBSCAN

def plot_data ():

    np.random.seed(0)

    # Generate datasets.
    n_samples = 500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axs = axs.flatten()

    # Plot dataset 1: Noisy Circles
    axs[0].scatter(noisy_circles[0][:, 0], noisy_circles[0][:, 1], s=15)
    axs[0].set_title('Noisy Circles')

    # Plot dataset 2: Noisy Moons
    axs[1].scatter(noisy_moons[0][:, 0], noisy_moons[0][:, 1], s=15)
    axs[1].set_title('Noisy Moons')

    # Plot dataset 3: Blobs
    axs[2].scatter(blobs[0][:, 0], blobs[0][:, 1], s=15)
    axs[2].set_title('Blobs')

    # Plot dataset 4: No Structure
    axs[3].scatter(no_structure[0][:, 0], no_structure[0][:, 1], s=15)
    axs[3].set_title('No Structure')

    # Plot dataset 5: Anisotropic
    axs[4].scatter(X_aniso[:, 0], X_aniso[:, 1], s=15)
    axs[4].set_title('Anisotropic')

    # Plot dataset 6: Varied Variances
    axs[5].scatter(varied[0][:, 0], varied[0][:, 1], s=15)
    axs[5].set_title('Varied Variances')

    # Remove ticks from all subplots
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(".\plots\plot_datasest.jpeg")
    img=cv2.imread(".\plots\plot_datasest.jpeg")
    res, im_png = cv2.imencode(".jpeg", img)
    return im_png
