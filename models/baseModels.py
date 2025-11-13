import math
from turtle import forward

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from utils.mykmeans import myKMeans
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils.graph_adjacency import *
from utils.util import *
from config import*

class ClusterLayer(nn.Module):
    """get the clustering centroids, soft labels, target distribution and labels"""

    def __init__(self, n_clustering, init_centers=None, max_iter=300):
        super(ClusterLayer, self).__init__()
        self._n_clustering = n_clustering
        self.device = device
        self._init_centers = init_centers
        self.centroids = None

    def forward(self, X, init_centroids=None):
        X_cpu = X.to('cpu')
        with torch.no_grad():
            if init_centroids is not None:
                if init_centroids.device != 'cpu':
                    init_centroids = init_centroids.to('cpu')
                self.cluster = KMeans(n_clusters=self._n_clustering, random_state=0, init=init_centroids,
                                      max_iter=1000).fit(X_cpu)
            else:
                self.cluster = KMeans(n_clusters=self._n_clustering, random_state=0, max_iter=1000).fit(X_cpu)
            self.centroids = self.cluster.cluster_centers_
            self.centroids = torch.tensor(self.centroids).float()

        if self.centroids.is_cpu:
            self.centroids = self.centroids.to(self.device)
        distances = torch.cdist(X, self.centroids)
        soft_labels = 1.0 / (distances ** 2 + 1)
        soft_labels = soft_labels / torch.sum(soft_labels, dim=1, keepdim=True)
        target_labels = (soft_labels ** 2) / (torch.sum(soft_labels ** 2, dim=1, keepdim=True))
        labels = torch.argmax(target_labels, dim=1)
        labels = labels.cpu().numpy()
        return self.centroids, soft_labels, target_labels, labels



class GraphSelfAttention(nn.Module):
    def __init__(self, hidden_size, output_dim=None):
        super(GraphSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim if output_dim is not None else hidden_size
        
        # 注意力权重计算
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 输出投影
        self.W_o = nn.Linear(hidden_size, self.output_dim)
        
    def forward(self, X, adj, return_attention=False):
        """
        前向传播
        Args:
            X: 输入特征矩阵 [N, D]
            adj: 邻接矩阵 [N, N]
            return_attention: 是否返回注意力权重
        Returns:
            output: 注意力输出 [N, D]
            attention: 注意力权重 [N, N] (可选)
        """

        Q = self.W_q(X)  # [N, D]
        K = self.W_k(X)  # [N, D]
        V = self.W_v(X)  # [N, D]
        
        attention = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_size)  # [N, N]
        
        attention = attention.masked_fill(adj == 0, float('-inf'))
        
        attention = F.softmax(attention, dim=-1)
        
        output = torch.matmul(attention, V)  # [N, D]
        output = self.W_o(output)
        
        if return_attention:
            return output, attention
        return output



class SingleViewGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim,num_layers=3):
        """
        Initializes the single-view graph attention encoder.
        Args:
            input_dim (int): The input dimension for the current view.
            hidden_size (int): The dimension of the hidden layer (or intermediate layer).
            num_layers (int): The number of encoder layers.
        """
        super(SingleViewGraphEncoder, self).__init__()
        self.num_layers = num_layers
        
        # 特征映射矩阵 (从输入维度到隐藏维度)
        self.projection = nn.Linear(input_dim, hidden_size)
        self.projection_bn=nn.BatchNorm1d(hidden_size)
        
        # 创建多层编码器
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            
            # 最后一层的输出维度为256，其他层为hidden_size
            current_out_dim = hidden_size if i != num_layers - 1 else output_dim
            attention_layer = GraphSelfAttention(hidden_size,current_out_dim)
            
            linear_layer = nn.Linear(hidden_size, current_out_dim)
            bn_layer = nn.BatchNorm1d(current_out_dim)
            activation_layer = nn.ReLU()
            self.layers.append(nn.ModuleDict({
                'attention': attention_layer,
                'linear': linear_layer,
                'batch_norm': bn_layer,
                'activation': activation_layer
            }))
    
    def get_embedding_features(self, X, adj_glo):
        """
        获取通过嵌入层得到的初级特征
                Obtains the initial features produced by the embedding layer.
        Args:
            X (torch.Tensor): The input feature matrix, typically of shape `[N, D_input]`.
        Returns:
            embed_features (torch.Tensor): The initial features output by the embedding layer, with shape `[N, hidden_size]`.
        """
        X=torch.matmul(normalize_adj(adj_glo),X)
        X=self.projection(X)
        X=self.projection_bn(X)
        return X
            
    def forward(self, X, adj,adj_glo):
        h = self.get_embedding_features(X,adj_glo) # [N, hidden_size]
        embed_features=h
        
        for i, layer in enumerate(self.layers):
            h_s=h
            attention_layer = layer['attention']
            batch_norm_layer = layer['batch_norm']
            activation_layer = layer['activation']
            h = attention_layer(h, adj) # [N, hidden_size]
            

            if i != self.num_layers - 1:
                h = batch_norm_layer(h) # [N, current_out_dim]
                h = activation_layer(h) # [N, current_out_dim]
                h = h+h_s
            else:
                h = batch_norm_layer(h) # [N, current_out_dim]
                h= activation_layer(h)

                
        return h,embed_features





        


