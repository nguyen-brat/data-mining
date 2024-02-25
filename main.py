import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

def generate_dataset(dataset_type):
    if dataset_type == 'Moons':
        return make_moons(n_samples=200, noise=0.05, random_state=42)
    elif dataset_type == 'Blobs':
        return make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)
    elif dataset_type == 'Circles':
        return make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)

def dbscan_demo(X, eps, min_samples):
    # Fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    X = StandardScaler().fit_transform(X)
    dbscan.fit(X)
    labels = dbscan.labels_

    # Plot clusters
    unique_labels = np.unique(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure(figsize=(12, 6))

    # Plot DBSCAN result
    plt.subplot(1, 2, 1)
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=10)
    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)

def kmeans_demo(X, n_clusters):
    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    X = StandardScaler().fit_transform(X)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Plot clusters
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)

def main():
    st.title('Clustering Algorithms Demo')

    # Select dataset type
    dataset_type = st.selectbox('Select Dataset Type', ['Moons', 'Blobs', 'Circles'])

    # Generate selected dataset
    X, _ = generate_dataset(dataset_type)

    # Parameters for DBSCAN
    eps = st.slider('EPS (DBSCAN)', min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    min_samples = st.slider('Min Samples (DBSCAN)', min_value=1, max_value=20, value=5)

    # Parameters for K-Means
    n_clusters = st.slider('Number of Clusters (K-Means)', min_value=2, max_value=10, value=3)

    # Run DBSCAN demo
    dbscan_demo(X, eps, min_samples)

    # Run K-Means demo
    kmeans_demo(X, n_clusters)

    plt.tight_layout()
    st.pyplot(plt)

if __name__ == "__main__":
    main()
