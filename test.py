import torch
import numpy as np
import pickle as pkl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import EmbeddingsTestDataset, PreTrainedDataset

from data import EmbeddingsTestDataset
from utils import compute_mAP, MLPResidualAdapter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(eval_dataset, model, ks=[1, 10, 50, 100, 200]):
    # pass the query features through the model:
    query_feats = []
    for query in eval_dataset.query_feats:
        query = query.to(device)
        query_feats.append(model(query))
    query_feats = torch.stack(query_feats, axis=0).cpu().detach()

    # pass the gallery features through the model:
    gallery_feats = []
    for gallery in eval_dataset.gallery_feats:
        gallery = gallery.to(device)
        gallery_feats.append(model(gallery))
    gallery_feats = torch.stack(gallery_feats, axis=0).cpu().detach()

    # get the mAP:
    return compute_mAP(query_feats, eval_dataset.query_classes,
               gallery_feats, eval_dataset.gallery_classes, 
               ks=ks)


if __name__ == '__main__':
    # restore the model:
    input_dim = 1024
    hidden_dims = [512, 512, 512]
    model = MLPResidualAdapter(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    ckpt = torch.load('checkpoint-ep-11-l-512-512-512.pth')
    model.load_state_dict(ckpt['model_state_dict'])

    # load the validation set in memory:
    val_data = PreTrainedDataset('D:\\image-image retrieval\\test_set_features_clipRN-50.pkl')
    #val_data = PreTrainedDataset('D:\\image-image retrieval\\validation_set_features_clipViT(B-16).pkl')

    mAP = evaluate_model(val_data, model)
    # mAP, rank = compute_mAP(val_data.query_feats, val_data.query_classes,
    #            val_data.gallery_feats, val_data.gallery_classes, 
    #            ks=[1, 10, 50, 100, 200])
    print('Overall mAP: ', mAP)
    # print('Mean Rank: ', rank)
