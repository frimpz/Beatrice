import csv
import random

import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def write_file(filename, dic):
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=':')
        for key, value in dic.items():
            writer.writerow([key, value])


def read_file(filename):
    with open(filename, 'r') as input_file:
        my_list = []
        types = str, str
        for line in input_file:
            elements = tuple(t(e) for t, e in zip(types, line.split()))
            my_list.append(elements)
        return my_list


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(adj, features=True):
    # adjacency matrix
    # adj = read_file(file_name)
    G = nx.from_numpy_matrix(adj)

    nodes = G.nodes()
    label = [i for i in range(len(nodes))]

    dictionary = dict(zip(nodes, label))
    G = nx.relabel_nodes(G, dictionary)
    # degrees = [i for i in G.degree()]
    # output = sorted(degrees, key=lambda x: x[-1], reverse=True)
    # print(output)
    write_file("rel/relabel-hs.csv", dictionary)
    # adj = nx.to_numpy_array(G)

    if features:
        pass
    else:
        features_1 = sp.identity(adj.shape[0]).tocsr()

    features = sparse_mx_to_torch_sparse_tensor(features_1)

    adj = scipy.sparse.csr_matrix(adj)
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    print("Finished data preprocessing")
    return adj, features


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.8).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def read_file2(filename):
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=':')
        my_dict = dict(reader)
        my_dict = {str(u): eval(v) for u, v in my_dict.items()}
        return my_dict


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.8).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
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
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]

    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 30.0))
    num_val = int(np.floor(edges.shape[0] / 10.0))


    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)


    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])


    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        # if ismember([idx_i, idx_j], train_edges):
        #     continue
        # if ismember([idx_j, idx_i], train_edges):
        #     continue
        # if ismember([idx_i, idx_j], val_edges):
        #     continue
        # if ismember([idx_j, idx_i], val_edges):
        #     continue
        # if ismember([idx_i, idx_j], edges_all):
        #     continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # pin = {(i, v) for i, v in test_edges}
    # pin_1 = {(i, v) for i, v in val_edges}
    # pin_2 = {(i, v) for i, v in edges_all}
    # print(pin_1.issubset(pin_2))
    # print(pin.issubset(pin_2))

    print(type(val_edges_false))
    print(type(val_edges_false[0]))
    print(test_edges_false)
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_edges2(adj, size):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]

    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 30))
    num_val = int(np.floor(edges.shape[0] / 10))

    edges_all = edges_all.tolist()
    edges_all = set(tuple(row) for row in edges_all)
    # print(len(set(edges_all)))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[0:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    val_edges = val_edges.tolist()
    val_edges = set(tuple(row) for row in val_edges)
    test_edges = test_edges.tolist()
    test_edges = set(tuple(row) for row in test_edges)

    false_test_edges = set()
    while len(false_test_edges) < len(test_edges):
        i = random.sample(range(0, adj.shape[0]), size)
        j = random.sample(range(0, adj.shape[0]), size)
        _edges = set([(a, b) for a, b in zip(i, j) if a != b])
        _edges = _edges.difference(edges_all)
        false_test_edges.update(_edges)
    # false_test_edges = list(false_test_edges)
    # false_test_edges = [[a, b] for a, b in false_test_edges]

    false_val_edges = set()
    while len(false_val_edges) < len(val_edges):
        i = random.sample(range(0, adj.shape[0]), size)
        j = random.sample(range(0, adj.shape[0]), size)
        _edges = set([(a, b) for a, b in zip(i, j) if a != b])
        _edges = _edges.difference(edges_all)
        false_val_edges.update(_edges)
    # false_val_edges = [[a, b] for a, b in false_val_edges]
    # false_val_edges = list(false_val_edges)

    # edges_all = np.array(edges_all)

    def ismember(a, b):
        return len(a.intersection(b)) == 0

    train_edges_as_set = train_edges.tolist()
    train_edges_as_set = set(tuple(row) for row in train_edges_as_set)
    assert ~ismember(false_test_edges, edges_all)
    assert ~ismember(false_val_edges, edges_all)
    assert ~ismember(val_edges, train_edges_as_set)
    assert ~ismember(test_edges, train_edges_as_set)
    assert ~ismember(val_edges, test_edges)
    assert ~ismember(false_val_edges, train_edges_as_set)
    assert ~ismember(false_val_edges, val_edges)

    false_val_edges = [[a, b] for a, b in false_val_edges]
    false_test_edges = [[a, b] for a, b in false_test_edges]

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, list(false_val_edges), test_edges, list(false_test_edges)



# print(check_symmetric(load_data(False)[0]))

# load_data(features=False)