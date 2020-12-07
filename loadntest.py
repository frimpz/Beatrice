import pickle
import time
import sys
import math
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import argparse
import networkx as nx
import pandas as pd
import plots
from model import GCNModelAE, GCNModelAE_UN
from plots import opt_dbScan, k_meansElbow
from utils import load_data, preprocess_graph, read_file2, mask_test_edges, get_roc_score
import numpy as np
from sklearn.manifold import TSNE


pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 50000)
pd.set_option('display.width', 1000)


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--ndim', type=int, default=16,
                    help='Number of dimension.')
parser.add_argument('--dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--saved-model', type=str, default='tr/sup-comb', help='Saved model')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def read_file(filename):
    with open(filename, 'r') as input_file:
        my_list = []
        types = str, str, str
        for line in input_file:
            elements = tuple(t(e) for t, e in zip(types, line.split()))
            my_list.append(elements)
        return my_list


def read_file_node_scores(filename):
    with open(filename, 'r') as input_file:
        next(input_file)
        scores = {}
        types = str, str, float
        for line in input_file:
            elements = tuple(t(e) for t, e in zip(types, line.split()))
            scores[elements[1]] = elements[2]
        return scores


def try_data(_prediction):
    final = torch.sigmoid(_prediction)
    return final


def plot_optimal_clusters(data, type):
    # Cluster with DB-Scan
    opt_dbScan(data, type)
    # Optimal Clusters K-Means
    k_meansElbow(data, type)
    sys.exit()


# Cluster with DB-Scan
# DB-Scan
def db_scan(data, eps, min_samples):
    dbsc = DBSCAN(eps=eps, min_samples=min_samples)
    dbs_model = dbsc.fit(data)
    pickle.dump(dbs_model, open("dbs_model.pkl", 'wb'))
    dbs_labels = dbsc.labels_
    return dbs_labels


# Cluster with K-means
# K-Means
def k_means(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans_model = kmeans.fit(data)
    pickle.dump(kmeans_model, open("kmeans_model.pkl", 'wb'))
    kmeans_labels = kmeans.labels_
    return kmeans_labels


def plots_pairwise(vars, dsc_unique_clusters, kmeans_unique_clusters):
    # vars is column names
    x_vars = vars[0:int(len(vars) / 2)]
    y_vars = vars[int(len(vars) / 2):]

    # Pairwise plot with DB Scan labels
    plots.pairwise_scatter(data=data_df, hue='DBS-clusters',
                           title="Pairwise Scatter Plot for DB-Scan Clustering",
                           x_vars=x_vars, y_vars=y_vars,
                           marker=list(range(0, len(dsc_unique_clusters))))

    # Pairwise plot with K-means labels
    plots.pairwise_scatter(data=data_df, hue='KMeans-clusters',
                           title="Pairwise Scatter Plot for KMeans Clustering",
                           x_vars=x_vars, y_vars=y_vars,
                           marker=list(range(0, len(kmeans_unique_clusters))))

    # Feature importance plot heatmap
    plots.heat_map(data_df.iloc[:, 0:16], "Correlation matrix for Node Embeddings")


def plot_tsne(data, pp):
    time_start = time.time()
    n_components = 2
    perplexity = pp
    n_iter = 400
    tsne = TSNE(n_components, perplexity, n_iter)
    tsne_results = tsne.fit_transform(data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    # Plot DB-Scan from t-SNE
    plots.tsne_plot(x=tsne_results[:, 0], y=tsne_results[:, 1],
                    clusters=data_df['DBS-clusters'],
                    title="Clustering node embeddings with DBSCAN, "
                          "Perplexity: {} -- Number of Iterations: {}".format(perplexity, n_iter))

    # Plot K-means  from t-SNE
    plots.tsne_plot(x=tsne_results[:, 0], y=tsne_results[:, 1],
                    clusters=data_df['KMeans-clusters'],
                    title="Clustering node embeddings with KMeans"
                          "Perplexity: {} -- Number of Iterations: {}".format(perplexity, n_iter))


# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)
#
# adj, features = load_data('graphs/MM-HS-Combined-01132020.tsv', False)
#
#
# # Store original adjacency matrix (without diagonal entries) for later
# adj_orig = adj
# adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
# adj_orig.eliminate_zeros()
#
#
# adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
# # adj = adj_train
#
# adj_train = adj
# adj_norm = preprocess_graph(adj)
# adj_label = adj_train + sp.eye(adj_train.shape[0])
# adj_label = torch.FloatTensor(adj_label.toarray())
#
#
# pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
# norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
#
#
# model = GCNModelAE(nfeat=features.shape[1],
#             nhid=args.hidden,
#             nclass=args.ndim,
#             dropout=args.dropout)
# model.load_state_dict(torch.load(args.saved_model))
# model.eval()
#
#
# model(features, adj_norm)
# output = model.mu.data
#
# roc_score, ap_score = get_roc_score(output, adj_orig, test_edges, test_edges_false)
# roc_curr1, ap_curr1 = get_roc_score(output, adj_orig, val_edges, val_edges_false)
# print('Test ROC score: ' + str(roc_score))
# print('Test AP score: ' + str(ap_score))
#
# print('Test ROC score: ' + str(roc_curr1))
# print('Test AP score: ' + str(ap_curr1))
#
#
#
# # Normalize the output data
# data = output
# #scaler = StandardScaler().fit(output)
# #data = scaler.transform(output)
#
# # Convert to pandas
# meta_df = {}
# vars = []
# for i in range(args.ndim):
#     meta_df[str(i)] = output[:, i]
#     vars.append(str(i))
# data_df = pd.DataFrame(meta_df)
#
#
# #data_df.to_pickle("ldntst/mm-go-new")
#
#
# # data_df = pd.read_pickle("ldntst/mm-go-new")
# # data = data_df.to_numpy()
#
# # Optimal Clusters
# # neigh -- 0.03
# # random --
# # plot_optimal_clusters(data, args.saved_model+" "+args.loss)
# dbs_labels = db_scan(data, 0.15, 29)
# kmeans_labels = k_means(data, 30)
#
# # Add labels
# data_df['DBS-clusters'] = dbs_labels
# data_df['KMeans-clusters'] = kmeans_labels
# dsc_unique_clusters = set(dbs_labels)
# kmeans_unique_clusters = set(kmeans_labels)
#
#
# scores = read_file_node_scores('komp-hs.tsv')
# res = read_file2("rel/relabel-hs-go.csv")
# _dict = dict((v, k) for k, v in res.items())
# data_df['gensets'] = data_df.index.map(_dict)
# data_df['scores'] = data_df.gensets.map(scores)


data_df = pd.read_pickle("saved_combined2-30")
data_df = data_df[['gensets', 'KMeans-clusters']]

k_m = data_df[['gensets', 'KMeans-clusters']]\
    .sort_values('KMeans-clusters')\
    .set_index(['KMeans-clusters', 'gensets'], inplace=False)


scored_genesets = read_file('komp.tsv')
scored_genesets = sorted(scored_genesets, key=lambda x: x[1])
scored_genesets = [i[1] for i in scored_genesets]

filterd_df = data_df.loc[data_df['gensets'].isin(scored_genesets)]


x = math.ceil(0.2*len(scored_genesets))
tp_20 = scored_genesets[0: x]
top_20_percent = data_df.loc[data_df['gensets'].isin(tp_20)]


kms_full = data_df.groupby('KMeans-clusters').agg({'gensets': 'count'})
kms_full['complete-list'] = kms_full.index
member_22 = filterd_df.groupby('KMeans-clusters').agg({'gensets': 'count'})
member_22['komp-list'] = member_22.index
_top_20_percent = top_20_percent.groupby('KMeans-clusters').agg({'gensets': 'count'})
_top_20_percent['top-20'] = _top_20_percent.index

members = pd.merge(kms_full, member_22, how='outer', left_on='complete-list', right_on='komp-list')\
    .merge(_top_20_percent, how='outer', left_on='complete-list', right_on='top-20')
members = members.drop(columns=['komp-list', 'top-20'])
members.rename(columns={'complete-list': 'Cluster', 'gensets_x': 'Complete-list', 'gensets_y': 'komp-list', 'gensets': 'top-20'}, inplace=True)
members = members[['Cluster', 'top-20', 'komp-list', 'Complete-list']]



writer = pd.ExcelWriter('Combinedd2.xlsx')
data_df.to_excel(writer, sheet_name='full')
filterd_df.to_excel(writer, sheet_name='comp')
members.to_excel(writer, sheet_name='member-counts-kmeans')
writer.save()

exit()


# writer = pd.ExcelWriter('db_s.xlsx')
# db_s.to_excel(writer)
# writer.save()
#
# writer = pd.ExcelWriter('k_m.xlsx')
# k_m.to_excel(writer)
# writer.save()



# Plots here
# plots_pairwise(vars, dsc_unique_clusters, kmeans_unique_clusters)

# plot_tsne(data_df.iloc[:, 0:16], 10)


# ids = get_geneset(None)
# indicies = get_indicies(ids)
# kmeans = pickle.load(open("kmeans_model.pkl", 'rb'))
#
#
# rankings = []
# rankings_to_df = []
# for i, j in zip(indicies, ids):
#     top_k = 50
#     point_embeddings = [data[i]]
#     cluster = kmeans.predict(point_embeddings)
#     # cluster_points = data_df.loc[data_df['KMeans-clusters']
#     #     .isin(cluster)][['0', '1', '2', '3', '4', '5',
#     #                      '6', '7', '8', '9', '10', '11',
#     #                      '12', '13', '14', '15']]
#
#     cluster_points = data_df[['0', '1', '2', '3', '4', '5',
#                          '6', '7', '8', '9', '10', '11',
#                          '12', '13', '14', '15']]
#
#
#     ind = cluster_points.index.values
#     i = ind.shape[0]
#     ind = ind.reshape((1, i)).tolist()[0]
#     gensets = get_geneset(ind)
#
#     pr = point_ranking(point_embeddings, cluster_points, ind, gensets, top_k)# [:top_k]
#     rankings.append(pr)
#     for k in pr:
#         zone = [j, ]
#         zone.append(k[1])
#         zone.append(k[2])
#         rankings_to_df.append(zone)
#
#     cols = ['Node', 'Similar', 'rank-distance']
#
#
# rank_df = pd.DataFrame(rankings_to_df, columns=cols)
# rank_df = rank_df.set_index(['Node', 'Similar'], inplace=False)
#
#
# writer = pd.ExcelWriter('Neigh.xlsx')
# rank_df.to_excel(writer)
# writer.save()

# print(rank_df)


# for i in rankings:
#     print(i)

# adj_df = pd.read_pickle("../data/adj_data/6.txt")
# res = read_file("../data/gene/geneset_pairing.csv")
# dict = dict((v, k) for k, v in res.items())
# adj_df['gensets'] = data_df.index.map(dict)
# adj_df.set_index('gensets', inplace=True)
# adj_df = adj_df.rename(columns=dict)
#
# # print(adj_df.loc[34667])
#
# my_data = [(1, 3, 5), (1, 2, 7), (2, 1, 5)]
# my_columns = ['a', 'b', 'c']
# my_df = pd.DataFrame(my_data, columns=my_columns)
# my_df = my_df.set_index(['a', 'b'], inplace=False)
# print(my_df)