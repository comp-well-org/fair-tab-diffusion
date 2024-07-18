import os
import torch
import json
import numpy as np
import pandas as pd
import skops.io as sio
from torch.utils.data import DataLoader, Dataset

class XYCTabDataset(Dataset):
    def __init__(self, features, cond):
        self.features = features
        self.cond = cond
        self.feature_matrix = torch.from_numpy(features.values).float()
        self.cond_matrix = torch.from_numpy(cond.values).float()
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.feature_matrix[idx], self.cond_matrix[idx]

class XYCTabDataModule:
    def __init__(self, root: str, batch_size: int) -> None:
        self.root = root
        self.batch_size = batch_size
        
    def get_norm_fn(self) -> callable:
        fn = sio.load(os.path.join(self.root, 'fn.skops'))
        return fn

    def get_data_description(self) -> dict:
        with open(os.path.join(self.root, 'desc.json'), 'r') as f:
            description = json.load(f)
        return description
    
    def get_dataloader(
        self, 
        flag: str,
        normalize: bool = True,
    ) -> DataLoader:
        assert flag in ['train', 'eval', 'test']
        if normalize:
            x_filename = f'xn_{flag}.csv'
        else:
            x_filename = f'x_{flag}.csv'
        y_filename = f'y_{flag}.csv'
        if flag == 'train':
            shuffle = True
        else:
            shuffle = False
        x_file = os.path.join(self.root, x_filename)
        y_file = os.path.join(self.root, y_filename)
        x = pd.read_csv(x_file, index_col=0)
        y = pd.read_csv(y_file, index_col=0)
        dataset = XYCTabDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader
    
    def get_empirical_dist(self) -> np.array:
        y_train = pd.read_csv(os.path.join(self.root, 'y_train.csv'), index_col=0).values
        answer = []
        for i in range(y_train.shape[1]):
            _, y_dist = torch.unique(torch.from_numpy(y_train[:, i]), return_counts=True)
            answer.append(y_dist.float())
        return answer
