# This file contains plots for visualizing data.
import time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve



# Pairwise Scatter plot for features
def heat_map(data, title):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f', linewidths=.05)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    hm.set_title(title, fontsize=14)
    plt.show()


def tsne_plot(x, y, clusters, title=""):
    df_plot = pd.DataFrame()
    df_plot['x'] = x
    df_plot['y'] = y
    df_plot['clusters'] = clusters
    # muted set1 hls husl
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71", "#BF3542",
              "#FAC386", "#C0AE65", "#B37358", "#ECDCAD", "#E49463", "#F33244", "#B69B95",
              "#17202A", "#641E16", "#C0392B", "#F5B7B1", "#8E44AD", "#4A235A", "#EAECEE",
              "#154360", "#17A589", "#145A32", "#F4D03F", "#7E5109", "#566573", "#D35400",
              "#E67E22", "#117A65", "#85C1E9", "#566573", "#0B5345", "#154360", "#73C6B6",
    ]
    flatui = flatui[:len(set(clusters))]
    color_palette = sns.color_palette("husl", len(set(clusters)))
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="x",
        y="y",
        hue=clusters,
        palette=color_palette,
        data=df_plot,
        legend="full",
        alpha=0.8
    )
    plt.title(title)
    plt.show()


def k_meansElbow(data, type):
    Sum_of_squared_distances = []
    _min = 10
    _max = 40
    stp = 5
    K = range(_min, _max, stp)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        Sum_of_squared_distances.append(km.inertia_)
    kn = KneeLocator(K, Sum_of_squared_distances, curve='convex', direction='decreasing')
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k --- '+ type)
    plt.text(_max - 2*stp, max(Sum_of_squared_distances), 'k = %d' % kn.knee)
    plt.show()


def plot_loss(x, y, x_label, y_label, title):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def auc_roc(emb, adj_orig, edges_pos, edges_neg, roc_score):
    lw = 2

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    ##########################confirm
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    fpr, tpr, tresholds = roc_curve(labels_all, preds_all)

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


