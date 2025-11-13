import math
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.util import *
from utils.graph_adjacency import *

EPS = sys.float_info.epsilon


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    def forward(self, q, p, reduction='sum'):
        
        log_p = torch.log(p)
        log_q = torch.log(q)
        kl_loss = torch.mean(p * (log_p - log_q))
            
        return kl_loss

def JS_Loss(q,p,reduction='mean'):
    kl_loss=KLLoss()
    m=(p+q)/2
    js_loss=1/2*(kl_loss(p,m,reduction='sum')+kl_loss(q,m,reduction='sum'))
    if reduction == 'mean':
        return js_loss/p.shape[0]
    else:
        return js_loss




class MultiViewGraphContrastiveLoss(nn.Module):
    """Multi-view graph contrastive loss function"""
    def __init__(self, temperature=0.5):
        super(MultiViewGraphContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_list, k=10):
        """
        Calculate the multi-view graph contrastive loss.
        Args:
            z_list: A list of all view embeddings, each element is [N, D_v]
        Returns:
            loss: Multi-view graph contrastive loss
        """
        num_views = len(z_list)
        if num_views < 2:
            return torch.tensor(0.0, device=z_list[0].device)

        total_loss = 0.0
        N = z_list[0].size(0)

        for i in range(num_views):
            for j in range(i + 1, num_views):
                z_v1=get_similarity_matrix(z_list[i])
                z_v2=get_similarity_matrix(z_list[j])


                h_pair = torch.cat((z_v1, z_v2), dim=0)
                
                h_pair = F.normalize(h_pair, dim=1)
                
                sim_matrix_pair = torch.mm(h_pair, h_pair.T) / self.temperature

                mask_no_self = torch.ones_like(sim_matrix_pair, dtype=torch.bool)
                mask_no_self.fill_diagonal_(False)

                positive_mask = torch.zeros_like(sim_matrix_pair, dtype=torch.bool)
                for k in range(N):
                    positive_mask[k, N + k] = True # z_v1[k] -> z_v2[k]
                    positive_mask[N + k, k] = True # z_v2[k] -> z_v1[k]
                
                numerator = torch.exp(sim_matrix_pair) * positive_mask
                numerator_sum = numerator.sum(dim=1) # (2N,)

                denominator = torch.exp(sim_matrix_pair) * mask_no_self
                denominator_sum = denominator.sum(dim=1) # (2N,)

                # InfoNCE Loss
                loss_per_anchor = -torch.log(numerator_sum / (denominator_sum + EPS) + EPS)
                
                total_loss += loss_per_anchor.sum()/N

        if num_views > 1:
            return total_loss / (num_views * (num_views - 1) / 2)
        else:
            return total_loss




