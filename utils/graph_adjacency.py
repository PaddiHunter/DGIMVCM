import numpy as np
import torch
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import torch.nn.functional as F

def pairwise_distance_between(data1, data2):
    """
    计算两组数据之间的欧氏距离
    Args:
        data1: 第一组数据 [N, D]
        data2: 第二组数据 [M, D] 
    Returns:
        dists: 距离矩阵 [N, M]
    """
    x_norm = torch.reshape(torch.sum(torch.square(data1), 1), [-1, 1])
    
    y_norm = torch.reshape(torch.sum(torch.square(data2), 1), [1, -1])
    
    dists = x_norm - 2 * torch.matmul(data1, data2.T) + y_norm
    
    return dists
def get_similarity_matrix_between(data1,data2,method='heat'):
    if method=='heat':
        dists=pairwise_distance_between(data1,data2)
        dists=torch.exp(-dists)
    return dists

def pairwise_distance(data):
    # 计算样本之间的欧氏距离
    x_norm = torch.reshape(torch.sum(torch.square(data), 1), [-1, 1])  # column vector
    x_norm2 = torch.reshape(torch.sum(torch.square(data), 1), [1, -1])  # column vector
    dists = x_norm - 2 * torch.matmul(data, data.T) + x_norm2
    return dists


def get_similarity_matrix(features, method='heat', norm=True):
    dist = None
    device = features.device
    if norm:
        if method=='heat':
            max_f, _ = torch.max(features, dim=0, keepdim=True)
            min_f, _ = torch.min(features, dim=0, keepdim=True)
            features = (features - min_f+1e-12) / (max_f - min_f+1e-12)
        if method=='cos':
            features = features / (torch.norm(features, dim=1, keepdim=True)+1e-12)
    if method == 'cos':

        dist = features @ features.T
    elif method == 'heat':
        dist = -0.5 * pairwise_distance(features)
        dist = torch.exp(dist)

    elif method == 'ncos':
        if features.is_cuda:
            features = features.to('cpu')
        features = normalize(features, axis=1, norm='l1')
        dist = features @ features.T
        dist = torch.from_numpy(dist)
        dist = dist.to('cuda')
    else:
        raise ValueError('Unknown method %s to get similarity matrix', method)

    return dist


# log checked
def get_masked_similarity_matrix(adj, k=10):
    """only keep k neighbor points that are most similar"""
    dist = adj.clone()
    topk = torch.topk(dist, k, dim=1)
    mask_matrix = torch.zeros((dist.shape[0], dist.shape[1]))
    index_row_vector = torch.tensor(range(dist.shape[0]))
    index_row_vector = index_row_vector.unsqueeze(dim=1)
    index_row_vector = index_row_vector.tile(1, k)
    index_row_vector = index_row_vector.flatten()
    index_col_vector = topk[1]
    index_col_vector = index_col_vector.flatten()
    mask_matrix[index_row_vector, index_col_vector] = 1
    mask_matrix = mask_matrix.bool().to(dist.device)
    mask_matrix = ~mask_matrix
    # dist[mask_matrix] = 0
    # return dist
    return torch.where(mask_matrix, torch.zeros_like(dist).to(dist.device), dist)


def get_masked_adjacency_matrix(adj, k=10):
    dist = adj.clone()
    topk = torch.topk(dist, k, dim=1)
    mask_matrix = torch.zeros((dist.shape[0], dist.shape[1]))
    index_row_vector = torch.tensor(range(dist.shape[0]))
    index_row_vector = index_row_vector.unsqueeze(dim=1)
    index_row_vector = index_row_vector.tile(1, k)
    index_row_vector = index_row_vector.flatten()
    index_col_vector = topk[1]
    index_col_vector = index_col_vector.flatten()
    mask_matrix[index_row_vector, index_col_vector] = 1
    mask_matrix = mask_matrix.bool()
    dist[mask_matrix] = 1
    mask_matrix = ~mask_matrix
    dist[mask_matrix] = 0
    return dist



def normalize_adj(adj):
    degree = torch.sum(adj, dim=1)
    d_hat = torch.diag(degree ** -0.5)
    adj = d_hat @ adj @ d_hat
    return adj
