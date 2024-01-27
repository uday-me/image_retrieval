import torch
from torch.utils.data import Dataset
import pickle as pkl


CLASS_NAMES = [
    'calling', 'dancing', 'fighting', 'listening_to_music',  'sleeping',
    'clapping', 'drinking', 'hugging', 'running', 'texting',
    'cycling', 'eating',  'laughing', 'sitting', 'using_laptop',
]
CLASS_NAMES = sorted(CLASS_NAMES)


class EmbeddingsDataset(Dataset):

    def __init__(self, is_train):
        self.data = []
        self.subset = 'train' if is_train else 'val'
        feature_dir = 'D:\\image-image retrieval\\data\\train' if self.subset == 'train' else 'D:\\image-image retrieval\\data\\val\\gallery'
        print(feature_dir)
        for i, c in enumerate(CLASS_NAMES):
            with open(f"{feature_dir}\\{c}.pkl", "rb") as f:
                d = pkl.load(f)
            for emb in d:
                emb = emb.squeeze(0)
                self.data.append((i, emb))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    

class EmbeddingsTestDataset(Dataset):

    def _read_pickle_files(self, pickle_dir):
        feats = []
        class_ids = []
        for i, c in enumerate(CLASS_NAMES):
            with open(f"{pickle_dir}\\{c}.pkl", "rb") as f:
                d = pkl.load(f)
            for emb in d:
                feats.append(emb.squeeze(0))
                class_ids.append(i)
        class_ids = torch.tensor(class_ids)
        feats = torch.tensor(feats)
        return class_ids, feats

    def __init__(self, split='val'):
        self.gallery_classes, self.gallery_feats = self._read_pickle_files(f"D:\\image-image retrieval\\data\\{split}\\gallery")
        self.query_classes, self.query_feats = self._read_pickle_files(f"D:\\image-image retrieval\\data\\{split}\\query")

    def __len__(self):
        return len(self.query_feats)

    def __getitem__(self, index):
        return self.query_classes[index], self.query_feats[index]

class PreTrainedDataset(Dataset):

    def _parse(self, feat_dict):
        feats = []
        class_ids = []
        for i, c in enumerate(CLASS_NAMES):
            feats.append(torch.tensor(feat_dict[c]).squeeze())
            class_ids.append(i * torch.ones(feat_dict[c].shape[0]))
        feats = torch.concat(feats, axis=0)
        class_ids = torch.concat(class_ids, axis=0)
        return class_ids, feats
        
    def _read_pickle_file(self, pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            q_feats, g_feats = pkl.load(f)
        self.query_classes, self.query_feats = self._parse(q_feats)
        self.gallery_classes, self.gallery_feats = self._parse(g_feats)

    def __init__(self, pickle_file_path):
        self._read_pickle_file(pickle_file_path)
    
    def __len__(self):
        return len(self.query_feats)
    
    def __getitem__(self, index):
        return self.query_classes[index], self.query_feats[index]