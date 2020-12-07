import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import numpy as np


def loss_function_ae(preds, labels, norm, pos_weight):
    pos_weight = np.array([pos_weight,], dtype=np.float64)
    pos_weight = torch.from_numpy(pos_weight)
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    return cost


def loss_function_vae(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    pos_weight = np.array([pos_weight,], dtype=np.float64)
    pos_weight = torch.from_numpy(pos_weight)
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD








