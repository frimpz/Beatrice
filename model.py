import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution


class GCNModelAE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNModelAE, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, dropout, act=F.relu)
        self.gc2 = GraphConvolution(nhid, nclass, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        self.mu = self.gc2(hidden1, adj)
        return self.mu

    def forward(self, x, adj):
        embedding = self.encode(x, adj)
        return self.dc(embedding)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.matmul(z, z.t()))
        return adj