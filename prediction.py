import pickle

import torch
import argparse
import pandas as pd
import numpy as np
import networkx as nx

from cluster_utils import plot_optimal_clusters, k_means, try_data, plot_tsne, plot_heat_map
from loss import loss_function_ae
from model import GCNModelAE
from plots import plot_loss, auc_roc
from utils import load_data, preprocess_graph, get_acc, mask_test_edges, get_roc_score, mask_test_edges2

# Saved models: gcn_model


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
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
parser.add_argument('--saved-model', type=str, default='tr/trained', help='Saved model')
parser.add_argument('--title', type=str, default=' -- G --', help='graph form')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

with open('test1.pkl', 'rb') as f:
    x = pickle.load(f)
    print(x.shape)
adj, features = load_data(x, False)

G = nx.from_numpy_matrix(adj.toarray())
adj_train = preprocess_graph(adj)


model = GCNModelAE(nfeat=features.shape[1],
                       nhid=args.hidden,
                       nclass=args.ndim,
                       dropout=args.dropout)
model.load_state_dict(torch.load(args.saved_model))
model.eval()


output = model(features, adj_train)
output = try_data(output)
output = output.detach().numpy()

# Normalize the output data
data = output
# scaler = StandardScaler().fit(output)
# data = scaler.transform(output)

# Convert to pandas
meta_df = {}
_vars = []
for i in range(args.ndim):
    meta_df[str(i)] = output[:, i]
    _vars.append(str(i))
data_df = pd.DataFrame(meta_df)

print(data_df)
plot_optimal_clusters(data, "K-means " + args.title)
kmeans_labels = k_means(data, 20, model_name="models/kmeans_gcn_model.pkl")

# Add labels
data_df['KMeans-clusters'] = kmeans_labels
kmeans_unique_clusters = set(kmeans_labels)

# Plots here
plot_heat_map(data_df.iloc[:, 0:16], "Correlation matrix for Node Embeddings " + args.title)

plot_tsne(data_df.iloc[:, 0:16], data_df['KMeans-clusters'],
          "Clustering node embeddings with KMeans Perplexity: {} -- ")

