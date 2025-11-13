import numpy as np
import torch.optim
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.manifold import TSNE

from utils.datasets import *
from models.baseModels import *
from torch.nn.functional import normalize
import torch.nn as nn
from utils.loss import *
from utils.util import *
from utils.evaluation import *
import matplotlib.pyplot as plt
from utils.visualization import *
from config import *
import torch.nn.functional as F


class MyModel(nn.Module):
    """多视图图神经网络模型"""
    def __init__(self, config):
        super(MyModel, self).__init__()
        self._config = config
        self._n_clusters = config['n_clustering']
        self._view_num = config['view_num']

        self._graph_rec_loss= config.get('graph_rec_loss_weight', 1.0)
        self._view_kl_loss_weight = config.get('view_kl_loss_weight', 1.0)
        self._graph_contrastive_loss_weight = config.get('graph_contrastive_loss_weight', 1.0)
        
        view_dims = self._config['view_dims']
        encoder_hidden_size = self._config['hidden_size']
        self.encoders=nn.ModuleList([
            SingleViewGraphEncoder(input_dim= view_dims[i], hidden_size=encoder_hidden_size, output_dim=self._config['output_dim'], num_layers=self._config['n_layers']) for i in range(self._view_num)
         ])
          
        self.cluster=ClusterLayer(n_clustering=self._config['n_clustering'])

    def run_train(self, X_list, mask, optimizer, Y_list, logger=None,ms=None):

        criterion_rec_g = nn.MSELoss(reduction='sum')
        criterion_kl_view = KLLoss()
        criterion_graph_contrastive = MultiViewGraphContrastiveLoss(temperature=self._config.get('graph_contrastive_loss_temperature', 0.5))
        
        # Aggregate the global graph
        centers=None
        raw_sim_v_list=[]
        for i in range(self._view_num):
                raw_sim_v = get_similarity_matrix(X_list[i])
                raw_sim_v[~mask[:, i].bool(),:]=0
                raw_sim_v[:,~mask[:, i].bool()]=0
                raw_sim_v_list.append(raw_sim_v)
        raw_sim_glo = sum(raw_sim_v_list) / len(raw_sim_v_list)
        adj_glo = get_masked_adjacency_matrix(raw_sim_glo, self._config['topk'])

        for epoch in range(self._config['training']["epoch"]):
            optimizer.zero_grad()
            loss = 0.0
            adj_v_list = []
            # primary features of embedding layer
            for i in range(self._view_num):
                with torch.no_grad():
                    embed_features = self.encoders[i].get_embedding_features(X_list[i],adj_glo)
                mask_v = mask[:, i].bool()
                sim_v = get_similarity_matrix(embed_features)
                adj_v = get_masked_adjacency_matrix(sim_v, self._config['topk'])                
                mask_mat = torch.zeros((mask_v.shape[0], mask_v.shape[0]), device=X_list[i].device)
                mask_mat[~mask_v, :] = 1
                mask_mat[:, ~mask_v] = 1 
                adj_v = torch.where(mask_mat.bool(), adj_glo, adj_v)
                adj_v_list.append(adj_v)

            # encoder
            z_list=[]
            embed_features_list=[]
            for i in range(self._view_num):
                z_v,embed_features=self.encoders[i](X_list[i], adj_v_list[i],adj_glo)
                z_list.append(z_v)
                embed_features_list.append(embed_features)
            
            # graph reconstruction loss
            for i in range(self._view_num):
                z_v = z_list[i]
                loss_rec = criterion_rec_g(get_masked_similarity_matrix(get_similarity_matrix(z_v), self._config['topk']), adj_glo)/adj_glo.shape[0]
                loss += self._graph_rec_loss * loss_rec
            
            # global clustering
            z_glo=torch.cat(z_list,dim=-1)
            centers, global_soft_labels, global_target_labels, y_pred = self.cluster(z_glo,centers)

            # clustering KL loss
            view_kl_loss = 0.0
            all_view_soft_labels = []

            # calculate view-specific current clustering centers and q_v (soft labels) for all epochs
            for i in range(self._view_num):
                z_v = z_list[i]
                
                # get view-specific current clustering centers from 'centers'
                encoder_hidden_size = z_v.shape[1]
                start_dim = i * encoder_hidden_size
                end_dim = (i + 1) * encoder_hidden_size
                view_specific_current_centroids = centers[:, start_dim:end_dim]
                
                # calculate soft labels (q_v) for the current view using Student-t distribution
                distance_to_centroids = torch.cdist(z_v, view_specific_current_centroids)
                q_v = 1.0 / (distance_to_centroids ** 2 + 1)
                # add epsilon to prevent division by zero and improve numerical stability
                q_v = q_v / (torch.sum(q_v, dim=1, keepdim=True))

                all_view_soft_labels.append(q_v)
                # calculate KL loss between view soft labels and global target distribution
                view_kl_loss += criterion_kl_view(q_v, global_target_labels)
            loss += self._view_kl_loss_weight * view_kl_loss

            # calculate graph contrastive loss
            graph_contrastive_loss = criterion_graph_contrastive(embed_features_list,self._config['topk'])
            loss += self._graph_contrastive_loss_weight * graph_contrastive_loss

            # calculate final clustering results using the average of all view soft labels
            if len(all_view_soft_labels) > 0:
                avg_view_soft_labels = torch.stack(all_view_soft_labels, dim=0).mean(dim=0)
                # calculate final y_pred using the average soft labels
                y_pred = torch.argmax(avg_view_soft_labels, dim=1).cpu().numpy()
            else:
                pass

            scores = evaluation(y_pred=y_pred, y_true=Y_list[0])
            if logger:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}, Scores: {scores}")
            else:
                if epoch % 10 == 0:
                    logging.info(f"Epoch {epoch}, Loss: {loss.item():.4f}, Scores: {scores}")
            
    
            loss.backward()
            optimizer.step()
                
            
        
        
        final_y_pred = y_pred
        


        return final_y_pred