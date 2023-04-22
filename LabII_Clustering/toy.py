from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import cv2
from sklearn import cluster, datasets, mixture
import numpy as np


#Creating toy data
def toy_data():
    return make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
    return_centers=True
    )

#plot kmeans clustering
def plot_some(n_clusters,k_means_labels,k_means_cluster_centers,X,k_medoids_labels,k_medoids_cluster_centers,y,distance_centers,distance_kmeans,distance_kmedoids):
    fig = plt.figure(figsize=(8,5))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ["#33ff00","#830095","#ff0055","#ffff00","#0026ff","#FF3E0E"]
    colors = colors[:n_clusters]

    #original
    ax = fig.add_subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="v", c=y)
    ax.set_title("Original data")
    ax.set_xticks(())
    ax.set_yticks(())
    for i in range(len(distance_centers)):
        plt.text(-13, 4.8 + i, "Dist 1 to "+str(-i+4)+" is: "+ str(round(distance_centers[i],2)))

    # KMeans
    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker="v")
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("KMeans")
    ax.set_xticks(())
    ax.set_yticks(())
    for i in range(len(distance_kmeans)):
        plt.text(-13, 4.8 + i, "Dist 1 to "+str(-i+4)+" is: "+ str(round(distance_kmeans[i],2)))

    # KMedoids
    ax = fig.add_subplot(1, 3, 3)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_medoids_labels == k
        cluster_center = k_medoids_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker="v")
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("KMedoids")
    ax.set_xticks(())
    ax.set_yticks(())
    for i in range(len(distance_kmedoids)):
        plt.text(-13, 4.8 + i, "Dist 1 to "+str(-i+4)+" is: "+ str(round(distance_kmedoids[i],2)))

    plt.savefig(".\plots\plot_globs.jpeg")
    img=cv2.imread(".\plots\plot_globs.jpeg")
    res, im_png = cv2.imencode(".jpeg", img)
    return im_png

def toy_data_2():
    n_samples = 500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None
    #Anisotropically distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    # blobs with varied variances
    varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )
    return [noisy_circles,noisy_moons,blobs,no_structure,aniso,varied]

def plot_toy_data_2():
    lista_labels_dat=["noisy_circles","noisy_moons","blobs","no_structure","aniso","varied"]
    lista_datos = toy_data_2()
    fig = plt.figure(figsize=(17, 5))
    for i in range(len(lista_datos)):
        x=lista_datos[i][0]
        y=lista_datos[i][1]
        ax = fig.add_subplot(1, len(lista_datos), i+1)
        ax.set_title("Plot "+lista_labels_dat[i])
        plt.scatter(x[:, 0], x[:, 1], marker="o", c=y)
        ax.set_xticks(())
        ax.set_yticks(())
    plt.savefig(".\plots\plot_dif_data.jpeg")
    img=cv2.imread(".\plots\plot_dif_data.jpeg")
    res, im_png = cv2.imencode(".jpeg", img)
    return im_png