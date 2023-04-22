import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
import cv2

#l2
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def silhouette_coefficient(data, labels, point_index):
    # Obtain the cluster label of the point
    point_label = labels[point_index]
    # Obtain the Euclidean distances from the point to the other points in its cluster and to the points in the other clusters.
    intra_cluster_distances = [euclidean_distance(data[point_index], data[i]) for i in range(len(data)) if labels[i] == point_label and i != point_index]
    inter_cluster_distances = [np.mean([euclidean_distance(data[point_index], data[i]) for i in range(len(data)) if labels[i] == j]) for j in np.unique(labels) if j != point_label]
    # Calculate the silhouette coefficient for the point
    if len(intra_cluster_distances) > 0 and len(inter_cluster_distances) > 0:
        a = np.mean(intra_cluster_distances)
        b = np.min(inter_cluster_distances)
        return (b - a) / np.max([a, b])
    else:
        return 0
    
def silhouette_score(data, labels):
    # Calculate the silhouette coefficient for each point in the data
    silhouette_scores = [silhouette_coefficient(data, labels, i) for i in range(len(data))]
    # Calculate the average silhouette coefficient for all points.
    return silhouette_scores,np.mean(silhouette_scores)

#To plot kmedoids silhouette
def silhouette_plot_kmeans(x_list,x_labels,n_plots):
    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.05, top=0.9)
    #plot silhouettes
    for j in range(n_plots):
        x=x_list[j]
        labels=x_labels[j]
        silhouette_vals = silhouette_samples(x, labels)
        silhouette_avg = np.mean(silhouette_vals)
        cluster_labels = np.unique(labels)
        n_clusters = cluster_labels.shape[0]
        silhouette_vals = [silhouette_vals[labels == i] for i in cluster_labels]
        ax = fig.add_subplot(2, n_plots, j+1)
        ax.set_title("score silhouette:" +str(round(silhouette_avg,3)))
        y_lower, y_upper = 0, 0
        yticks = []
        #colors
        colors = ["#5DADE2","#76D7C4","#F7DC6F","#BB8FCE","#F1948A","#BFC9CA"]
        colors = colors[:n_clusters]
        #silhouetee
        for i, cluster in enumerate(cluster_labels):
            cluster_silhouette_vals = silhouette_vals[i]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0,color=colors[i])
            ax.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(i+1))
            yticks.append((y_lower + y_upper) / 2.)
            y_lower += len(cluster_silhouette_vals)
        ax.axvline(silhouette_avg, color="red", linestyle="--")
        ax.set_xlabel("Coeficiente de silueta")
        ax.set_ylabel("Cluster")
        ax.set_yticks(())
        fig.tight_layout()

    #plot data
    n_clusters = cluster_labels.shape[0]
    colors = ["#5DADE2","#76D7C4","#F7DC6F","#BB8FCE","#F1948A","#BFC9CA"]
    colors = colors[:n_clusters]
    for l in range(n_plots):
        ax = fig.add_subplot(2, n_plots, l+n_plots+1)
        ax.set_title("Kmeans Cluster with k="+str(l+2))
        fig.tight_layout()
        for k, col in zip(range(l+2), colors):
            x=x_list[l]
            labels=x_labels[l]
            my_members = labels == k
            ax.plot(x[my_members, 0], x[my_members, 1], "w", markerfacecolor=col, marker="o")
            ax.set_xticks(())
            ax.set_yticks(())

    plt.savefig(".\plots\kmeans_clusters.jpeg")
    img=cv2.imread(".\plots\kmeans_clusters.jpeg")
    res, im_png = cv2.imencode(".jpeg", img)
    return im_png

#To plot kmedoids silhouette
def silhouette_plot_kmedoids(x_list,x_labels,n_plots):

    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.05, top=0.9)
    #plot silhouettes
    for j in range(n_plots):
        x=x_list[j]
        labels=x_labels[j]
        silhouette_vals = silhouette_samples(x, labels)
        silhouette_avg = np.mean(silhouette_vals)
        cluster_labels = np.unique(labels)
        n_clusters = cluster_labels.shape[0]
        silhouette_vals = [silhouette_vals[labels == i] for i in cluster_labels]
        ax = fig.add_subplot(2, n_plots, j+1)
        ax.set_title("score silhouette:" +str(round(silhouette_avg,3)))
        
        y_lower, y_upper = 0, 0
        yticks = []
        #colors
        colors = ["#5DADE2","#76D7C4","#F7DC6F","#BB8FCE","#F1948A","#BFC9CA"]
        colors = colors[:n_clusters]
        #silhouetee
        for i, cluster in enumerate(cluster_labels):
            cluster_silhouette_vals = silhouette_vals[i]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0,color=colors[i])
            ax.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(i+1))
            yticks.append((y_lower + y_upper) / 2.)
            y_lower += len(cluster_silhouette_vals)
            
        ax.axvline(silhouette_avg, color="red", linestyle="--")
        ax.set_xlabel("Coeficiente de silueta")
        ax.set_ylabel("Cluster")
        ax.set_yticks(())

    #plot data
    n_clusters = cluster_labels.shape[0]
    colors = ["#5DADE2","#76D7C4","#F7DC6F","#BB8FCE","#F1948A","#BFC9CA"]
    colors = colors[:n_clusters]
    for l in range(n_plots):
        ax = fig.add_subplot(2, n_plots, l+n_plots+1)
        ax.set_title("Kmedoids Cluster with k="+str(l+2))
        fig.tight_layout()
        for k, col in zip(range(l+2), colors):
            x=x_list[l]
            labels=x_labels[l]
            my_members = labels == k
            ax.plot(x[my_members, 0], x[my_members, 1], "w", markerfacecolor=col, marker="o")
            ax.set_xticks(())
            ax.set_yticks(())

    plt.savefig(".\plots\kmedoids_clusters.jpeg")
    img=cv2.imread(".\plots\kmedoids_clusters.jpeg")
    res, im_png = cv2.imencode(".jpeg", img)
    return im_png

def best_score_silhouette(x_list,x_labels,n_plots):
    best_sil = []
    for j in range(n_plots):
        x=x_list[j]
        labels=x_labels[j]
        silhouette_vals = silhouette_samples(x, labels)
        silhouette_avg = np.mean(silhouette_vals)
        best_sil.append(silhouette_avg)
    print(best_sil)
    return max(best_sil), best_sil.index(max(best_sil))+2