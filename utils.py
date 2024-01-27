import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import sklearn.metrics
import torch.nn.functional as F


def get_mAP(feats, labels):
    """mAP for a training batch.
    
    feats: [B, D] shape.
    labels: [B] shape.
    """
    feats = F.normalize(feats, dim=-1, p=2)
    sims = torch.matmul(feats, feats.T)
    gt_sims = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
    sims = sims.cpu().detach().numpy()
    gt_sims = gt_sims.cpu().detach().numpy()
    aps = []
    for i in range(feats.shape[0]):
       ap = sklearn.metrics.average_precision_score(gt_sims[i], sims[i])
       aps.append(ap)
    return np.mean(aps)


def compute_mAP(query_feats, query_labels,
               gallery_feats, gallery_labels, 
               ks=None):
    """mAP@k for given query and gallery data.
    
    query_feats: [N, D] shape.
    query_labels: [N] shape.
    gallery_feats: [M, D] shape.
    gallery_labels: [M] shape.
    ks: list of scalars or None.
    """
    query_feats = F.normalize(query_feats, dim=-1, p=2)
    gallery_feats = F.normalize(gallery_feats, dim=-1, p=2)

    sims = torch.matmul(query_feats, gallery_feats.T)
    gt_sims = torch.eq(query_labels.view(-1, 1), gallery_labels.view(1, -1)).float()

    rank_idx = torch.argsort(-sims, axis=-1)
    sorted_labels = torch.gather(gt_sims, axis=1, index=rank_idx).float()
    # first hit: index of first non-zero value in sorted_labels:
    first_hit_rank = torch.argmax(sorted_labels * torch.arange(sorted_labels.shape[1], 0, -1), axis=1)
    mean_rank = torch.mean((1 + first_hit_rank.float()))

    idx = 1 + torch.arange(sorted_labels.shape[1])
    precision = torch.cumsum(sorted_labels, dim=1) / idx.view(1, -1)
    rel_precision = sorted_labels * precision

    if ks is None:
      AP = rel_precision.sum(axis=1) / torch.clamp(sorted_labels.sum(axis=1), min=1.0)
      mAP = torch.mean(AP)
      return mAP
    else:
        mAP_at_k = {}
        for k in ks:
            if k <= 0 or k > gallery_feats.shape[0]:
               raise ValueError(f'Invalid value of k = {k}.')
            AP_at_k = rel_precision[:, :k].sum(axis=1) / torch.clamp(sorted_labels[:, :k].sum(axis=1), min=1.0)
            mAP_at_k[k] = torch.mean(AP_at_k)
        return mAP_at_k, mean_rank


class MLPResidualAdapter(nn.Module):
    """Residual MLP model."""
    def __init__(self, input_dim, hidden_dims, dropout=0.0):
        super(MLPResidualAdapter, self).__init__()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
           self.dropout = None
        self.fc_layers = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(input_dim)
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, input_dim)
    
    def forward(self, x):
        x  = x.float()
        x = self.layer_norm(x)
        residual = x
        for fc_layer in self.fc_layers:
            x = F.relu(fc_layer(x))
            if self.dropout:
               x = self.dropout(x)
        output = self.output_layer(x)
        output = output + residual
        return output


def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


class MultiPosConLoss(nn.Module):
  """
  Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
  ref: https://github.com/google-research/syn-rep-learn/blob/main/StableRep/models/losses.py#L49

  The code is adapted for local/non-distributed training, and simplified.
  """

  def __init__(self, temperature=0.1):
    super(MultiPosConLoss, self).__init__()
    self.temperature = temperature

  def set_temperature(self, temp=0.1):
    self.temperature = temp

  def forward(self, feats, labels):
    """
    feats shape: [B, D]
    labels shape: [B]
    """
    device = (torch.device('cuda')
              if feats.is_cuda
              else torch.device('cpu'))

    feats = F.normalize(feats, dim=-1, p=2)

    # Compute the mask based on labels
    labels = labels.float()
    is_correct = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(device)

    # Compute logits
    logits = torch.matmul(feats, feats.T) / self.temperature

    # Optional: subtract the largest logit to stabilize logits
    logits = stablize_logits(logits)

    # Compute ground-truth distribution
    p = is_correct / is_correct.sum(1, keepdim=True).clamp(min=1.0)
    loss = compute_cross_entropy(p, logits)
    return {'loss': loss, 'image_loss': loss}
