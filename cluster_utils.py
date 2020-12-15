import pickle
import sys
import time

import torch
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE

from plots import  k_meansElbow, heat_map, tsne_plot


def try_data(_prediction):
    final = torch.sigmoid(_prediction)
    return final


def plot_optimal_clusters(data, type):
    # Optimal Clusters K-Means
    k_meansElbow(data, type)
    sys.exit()





# Cluster with K-means
# K-Means
# model_name -> "kmeans_model.pkl"
def k_means(data, n_clusters, model_name):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans_model = kmeans.fit(data)
    pickle.dump(kmeans_model, open(model_name, 'wb'))
    kmeans_labels = kmeans.labels_
    return kmeans_labels


def plots_pairwise(data, hue, _vars, unique_clusters, title):
    # vars is column names
    x_vars = _vars[0:int(len(_vars) / 2)]
    y_vars = _vars[int(len(_vars) / 2):]

    # Pairwise plot with DB Scan labels
    pairwise_scatter(data=data, hue=hue, title=title,
                     x_vars=x_vars, y_vars=y_vars,
                     marker=list(range(0, len(unique_clusters))))


def plot_heat_map(data, title="Correlation matrix for Node Embeddings"):
    # Feature importance plot heatmap
    heat_map(data.iloc[:, 0:16], title)


def plot_tsne(data, clusters, title, perplexity, n_iter):
    time_start = time.time()
    n_components = 2
    perplexity = perplexity
    n_iter = n_iter
    tsne = TSNE(n_components, perplexity, n_iter, learning_rate=0.1)
    tsne_results = tsne.fit_transform(data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    # Plot DB-Scan or K-means from t-SNE
    tsne_plot(x=tsne_results[:, 0], y=tsne_results[:, 1],
              clusters=clusters,
              title=title.format(perplexity, n_iter))