import itertools
import random
import networkx as nx
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import numpy as np


class TripletLoss:

    """
    Class to calculate triplet loss

    Args:
        adj : adjacency matrix of graph [batch_size, batch_size]
        random_walk : find neighbours with random walk or connected nodes

    """

    def __init__(self, G, random_walk=True):
        super(TripletLoss, self).__init__()
        self.MARGIN = 3
        self.NUM_WALKS = 6
        self.POS_WALK_LEN = 1
        self.NEG_WALK_LEN = 8
        self.NUM_NEG = 6
        self.RDM_WALK = random_walk

        self.graph = G
        self.adj_lists = self.get_adjacency_list(G)
        self.nodes = list(self.adj_lists.keys())

        self._pos = []
        self._neg = []

    def get_adjacency_list(self, G):
        adj_list = {}
        for i in G.adjacency():
            adj_list[i[0]] = [int(x) for x in i[1] if int(x) is not i[0]]
        return adj_list

    # Defines triplet loss
    def pairwise_distances(self, embeddings, squared=False):
        # get dot product (batch_size, batch_size)
        dot_product = torch.matmul(embeddings, embeddings.t())

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        square_sum = torch.diag(dot_product)
        distances = square_sum.unsqueeze(1) - 2 * dot_product + square_sum.unsqueeze(0)
        distances = distances.clamp(min=0)

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            epsilon = 1e-16
            mask = torch.eq(distances, 0).float()
            distances += mask * epsilon
            distances = torch.pow(distances, 0.5)
            distances *= (1 - mask)
        return distances

    def get_random_walk(self):
        x = {}
        for node in self.nodes:
            for i in range(self.NUM_WALKS - 1):
                temp = list(self.graph.neighbors(node))
                # Remove co-occurrences
                temp = list(set(temp) - set([node]))
                if len(temp) == 0:
                    x[node] = set((node,))
                    break
                random_node = random.choice(temp)
                if node in x:
                    x[node].add(random_node)
                else:
                    x[node] = set((random_node,))
                node = random_node
        return x

    def get_neighbours_sorted_by_weight(self):
        x = {}
        for node in self.nodes:
            temp = [x[0] for x in sorted([(i, j['weight']) for i, j in dict(self.graph[node]).items()], key=lambda x: x[-1], reverse=True)]
            # Remove co-occurrences
            temp = list(set(temp) - set([node]))
            if len(temp) == 0:
                x[node] = set((node,))
            elif len(temp) < self.NUM_WALKS - 1:
                x[node] = set(temp)
            else:
                x[node] = set(temp[0:self.NUM_WALKS])
        return x

    def get_negtive_nodes(self):
        x = {}
        num_neg = self.NUM_NEG
        for node in self.nodes:
            neighbors = set([node])
            frontier = set([node])
            for i in range(self.NEG_WALK_LEN):
                current = set()
                for outer in frontier:
                    current |= set(self.adj_lists[outer])
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(self.adj_lists) - neighbors
            neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
            x[node] = set(neg_samples)
        return x

    def _get_anchor_positive_triplet_mask(self):

        """
        Returns a 2-D mask where mask[a, p] is True iff a and p are distinct and reachable in a random walk/ neighbours.

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]

        """

        size = len(self.nodes)
        mask = np.zeros((size, size))


        if self.RDM_WALK is False:
            pos = self.get_neighbours_sorted_by_weight()
        else:
            pos = self.get_random_walk()

        for node in pos:
            for neigh in pos[node]:
                mask[node, neigh] = True
                self._pos.append((node, neigh))

        return torch.BoolTensor(mask)

    def _get_anchor_negative_triplet_mask(self):

        """
        Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """

        size = len(self.nodes)
        mask = np.zeros((size, size))

        neg = self.get_negtive_nodes()
        for node in neg:
            for neigh in neg[node]:
                mask[node, neigh] = True
                self._neg.append((node, neigh))

        return torch.BoolTensor(mask)

    def _get_triplet_mask(self):

        """
        Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """

        size = len(self.nodes)
        triplets = np.zeros((size, size, size))

        for i in self._pos:
            for j in self._neg:
                if i[0] == j[0] and i[1] != j[1]:
                    triplets[i[0], i[1], j[1]] = True

        return torch.BoolTensor(triplets)

    def _batch_all_triplet_loss(self, embeddings, squared=False):

        """
        Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """

        pairwise_dist = self.pairwise_distances(embeddings, squared)

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)

        anchor_negative_dist = pairwise_dist.unsqueeze(1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.MARGIN

        mask = self._get_triplet_mask()

        triplet_loss = triplet_loss * mask.float()
        triplet_loss.clamp_(min=0)

        epsilon = 1e-16
        num_positive_triplets = (triplet_loss > 0).float().sum()
        num_valid_triplets = mask.float().sum()
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + epsilon)

        triplet_loss = triplet_loss.sum() / (num_positive_triplets + epsilon)

        return triplet_loss, fraction_positive_triplets

    def _batch_hard_triplet_loss(self, embeddings, squared=False):

        """
        Build the triplet loss over a batch of embeddings.
           For each anchor, we get the hardest positive and hardest negative to form a triplet.

           Args:
               labels: labels of the batch, of size (batch_size,)
               embeddings: tensor of shape (batch_size, embed_dim)
               margin: margin for triplet loss
               squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                        If false, output is the pairwise euclidean distance matrix.

        Returns:
           triplet_loss: scalar tensor containing the triplet loss
        """

        pairwise_dist = self.pairwise_distances(embeddings, squared)

        mask_anchor_positive = self._get_anchor_positive_triplet_mask()
        hardest_positive_dist = (pairwise_dist * mask_anchor_positive.float()).max(dim=1)[0]

        mask_negative = self._get_anchor_negative_triplet_mask()
        max_negative_dist = pairwise_dist.max(dim=1, keepdim=True)[0]
        distances = pairwise_dist + max_negative_dist * (~mask_negative).float()
        hardest_negative_dist = distances.min(dim=1)[0]

        triplet_loss = (hardest_positive_dist - hardest_negative_dist + self.MARGIN).clamp(min=0)
        triplet_loss = triplet_loss.mean()

        return triplet_loss



