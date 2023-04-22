import numpy as np
import random
from typing import Union
from fastapi import FastAPI
from fastapi.responses import FileResponse
import cv2
from cv2 import imread
from starlette.responses import StreamingResponse
import io
import matplotlib
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_blobs
import Unsupervised.kmeans as kmeans
import Unsupervised.kmedoids as kmedoids
import silhouette_2 as silhouette

from  datasets import plot_data
from  Clustering_SL import data_comp

from data_random import BlobPlotter
from toy import toy_data,plot_some,plot_toy_data_2

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/5_b_make_toyData")
def get_plot():
    bp = BlobPlotter(n_samples=500, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True, random_state=1)
    bp.generate_data()
    ime_png = BlobPlotter.plot_data(bp)
    return StreamingResponse(io.BytesIO(ime_png.tobytes()), media_type="image/jpeg")

@app.get("/5_c_Implementation_kmeans_kmedoids")
def cluster_distances():
    k=4
    x,y,centers = toy_data()
    np.random.seed(2)
    centroids,labels_kmeans  = kmeans.KMeans_2(K=k).fit(x)
    medoids,labels_kmedoids  = kmedoids.KMedoids_2(n_clusters=k).fit(x)
    distance_centers=[]
    distance_kmeans=[]
    distance_kmedoids=[]
    for i in range(len(centroids)-1):
        distance_centers.append(np.linalg.norm(centers[0] - centers[i+1]))
        distance_kmeans.append(np.linalg.norm(centroids[0] - centroids[i+1]))
        distance_kmedoids.append(np.linalg.norm(medoids[0] - medoids[i+1]))
    im_png=plot_some(k,labels_kmeans,centroids,x,labels_kmedoids,medoids,y,distance_centers,distance_kmeans,distance_kmedoids)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

@app.get("/5_c_Silhouette_kmeans")
def kmeans_silhouette(K= 5):
    x_list=[]
    x_labels=[]
    n_test=int(K)
    for i in range(n_test):
        k=i+2
        x,y,centers = toy_data()
        np.random.seed(2)
        k_means = kmeans.KMeans_2(K=k)
        centroids,cluster_labels  = k_means.fit(x)
        labels = k_means.predict(x)
        x_list.append(x)
        x_labels.append(labels)
    im_png = silhouette.silhouette_plot_kmeans(x_list,x_labels,n_test)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")


@app.get("/5_c_Silhouette_kmedoids")
def kmedoids_silhouette(K = 5):
    x_list=[]
    x_labels=[]
    n_test=int(K)
    for i in range(n_test):
        k=i+2
        x,y,centers = toy_data()
        np.random.seed(2)
        k_medoids = kmedoids.KMedoids_2(n_clusters=k)
        centroids,cluster_labels  = k_medoids.fit(x)
        labels = k_medoids.predict(x)
        x_list.append(x)
        x_labels.append(labels)
    im_png = silhouette.silhouette_plot_kmeans(x_list,x_labels,n_test)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

@app.get("/5_d_Silhouette_Score")
def score_silhouette(K = 5):
    x_list=[]
    x_labels=[]
    n_test=int(K)
    for i in range(n_test):
        k=i+2
        x,y,centers = toy_data()
        np.random.seed(2)
        k_means = kmeans.KMeans_2(K=k)
        centroids,cluster_labels  = k_means.fit(x)
        labels = k_means.predict(x)
        x_list.append(x)
        x_labels.append(labels)
    best_sil_kmeans,n_clus_means = silhouette.best_score_silhouette(x_list,x_labels,n_test)

    x_list=[]
    x_labels=[]
    n_test=int(K)  
    for i in range(n_test):
        k=i+2
        x,y,centers = toy_data()
        np.random.seed(2)
        k_medoids = kmedoids.KMedoids_2(n_clusters=k)
        centroids,cluster_labels  = k_medoids.fit(x)
        labels = k_medoids.predict(x)
        x_list.append(x)
        x_labels.append(labels)

    best_sil_kmedoids, n_clus_medoids = silhouette.best_score_silhouette(x_list,x_labels,n_test)
    return {"the best silhouette score in kmeans is " +str(best_sil_kmeans)+ " for clusters" : n_clus_means,
            "the best silhouette score in kmedoids is  "+str(best_sil_kmedoids)+ " for clusters" : n_clus_medoids}


@app.get("/6_a_Plot_datasets")
def Plot_datasets ():
    clus = plot_data()
    return StreamingResponse(io.BytesIO(clus.tobytes()), media_type="image/jpeg")

@app.get("/6_b_Apply_Clustering_SL")
def Apply_Clustering_SL ():
    clus = data_comp()
    return StreamingResponse(io.BytesIO(clus.tobytes()), media_type="image/jpeg")