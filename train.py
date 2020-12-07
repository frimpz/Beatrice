from __future__ import division
from __future__ import print_function

import pickle
import time
import argparse
import numpy as np
import torch
import scipy.sparse as sp
import torch.optim as optim

# Training settings

from loss import loss_function_ae
from model import GCNModelAE
from plots import plot_loss, auc_roc
from utils import load_data, preprocess_graph, get_acc, mask_test_edges, get_roc_score, mask_test_edges2

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
parser.add_argument('--nclass', type=int, default=16,
                    help='Number of clusters.')
parser.add_argument('--dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')

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


# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()


# adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges2(adj, adj.shape[0])
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

adj = adj_train

adj_norm = preprocess_graph(adj)
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = torch.FloatTensor(adj_label.toarray())


pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


# Model and optimizer
model = GCNModelAE(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=args.nclass,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


indices = []
losses = []
def train(epoch):
    with torch.autograd.set_detect_anomaly(True):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_norm)

        loss = loss_function_ae(preds=output, labels=adj_label, norm=norm, pos_weight=pos_weight)

        losses.append(loss)
        indices.append(epoch)
        loss.backward()
        curr_loss = loss.item()
        optimizer.step()

        acc_train = get_acc(output, adj_label)

        hidden_emb = model.mu.data.numpy()
        roc_curr, ap_curr= get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(curr_loss),
              'acc_train: {:.4f}'.format(acc_train),
              "val_ap=", "{:.5f}".format(ap_curr),
              "val_roc=", "{:.5f}".format(roc_curr),
              'time: {:.4f}s'.format(time.time() - t))


for epoch in range(args.epochs):
    train(epoch)


hidden_emb = model.mu.data.numpy()
roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))

auc_roc(hidden_emb, adj_orig, test_edges, test_edges_false, roc_score)


torch.save(model.state_dict(), "tr/trained")

# plot_loss(indices, losses, "Epooch", "Loss", "Epooch vs loss")
