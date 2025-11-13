import torch
import torch.nn.functional as F
import numpy as np

class myKMeans:
    def __init__(self, n_clusters, init_centers=None,max_iter=1000, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_=init_centers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    def fit_data(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # 随机初始化聚类中心
        if self.cluster_centers_ is None:
            random_indices = torch.randperm(self.n_samples)[:self.n_clusters]
            self.cluster_centers_ = X[random_indices]

        for i in range(self.max_iter):
            # 计算每个样本到聚类中心的距离, 并分配到最近的簇
            if self.cluster_centers_.is_cpu:
                self.cluster_centers_=self.cluster_centers_.to(device=self.device)
            distances = torch.cdist(X, self.cluster_centers_, p=2)
            self.labels_ = torch.argmin(distances, dim=1)

            # 计算新的聚类中心
            new_centers = torch.zeros_like(self.cluster_centers_)
            for j in range(self.n_clusters):
                cluster_points = X[self.labels_ == j]
                if cluster_points.shape[0] > 0:
                    new_centers[j] = cluster_points.mean(dim=0)
                else:
                    new_centers[j] = X[torch.randint(0, self.n_samples, (1,))]

            # 检查是否收敛
            center_shift = torch.norm(self.cluster_centers_ - new_centers)
            if center_shift <= self.tol:
                break

            self.cluster_centers_ = new_centers

        return self.cluster_centers_

    def predict_data(self, X):
        distances = torch.cdist(X, self.cluster_centers_, p=2)
        return torch.argmin(distances, dim=1)