import os
import torch
import json
import numpy as np
import skops.io as sio
from torch.utils.data import DataLoader, Dataset

class XYCTabDataset(Dataset):
    def __init__(self, features: np.ndarray, conds: np.ndarray, transforms: callable = None) -> None:
        """Initialize.
        
        Args:
            features: features of shape `(n, dx)`
            conds: conditions of shape `(n, dc)`
            transforms: transforms to apply to features
        
        NOTE:
            the first column of conds is identical to labels
        """
        self.features = torch.from_numpy(features).float()
        self.conds = torch.from_numpy(conds).float()
        self.transforms = transforms
    
    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item at index idx.
        
        Args:
            idx: index of the item
        
        Returns:
            tuple of features of shape `(dx,)` and  condition of shape `(dc,)` at index `idx`.
        """
        features = self.features[idx]
        conds = self.conds[idx]
        if self.transforms:
            features = self.transforms(features)
        return features, conds

class XYCTabDataModule:    
    def __init__(self, root: str, batch_size: int) -> None:
        """Initialize.
        
        Args:
            root: root directory of the dataset.
            batch_size: batch size.
        """
        self.root = root
        self.batch_size = batch_size
    
    def get_norm_fn(self) -> callable:
        fn = sio.load(os.path.join(self.root, 'fn.skops'))
        return fn
    
    def get_data_description(self) -> dict:
        with open(os.path.join(self.root, 'desc.json'), 'r') as f:
            description = json.load(f)
        return description

    def get_train_loader(self, shuffle: bool = True) -> DataLoader:
        xn_train = np.load(os.path.join(self.root, 'xn-train.npy'))
        y_train = np.load(os.path.join(self.root, 'y-train.npy'))
        train_set = XYCTabDataset(xn_train, y_train)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=shuffle)
        return train_loader
    
    def get_val_loader(self, shuffle: bool = False) -> DataLoader:
        xn_val = np.load(os.path.join(self.root, 'xn-val.npy'))
        y_val = np.load(os.path.join(self.root, 'y-val.npy'))
        val_set = XYCTabDataset(xn_val, y_val)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=shuffle)
        return val_loader

    def get_test_loader(self, shuffle: bool = False) -> DataLoader:
        xn_test = np.load(os.path.join(self.root, 'xn-test.npy'))
        y_test = np.load(os.path.join(self.root, 'y-test.npy'))
        test_set = XYCTabDataset(xn_test, y_test)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=shuffle)
        return test_loader
    
    def get_empirical_dist(self) -> np.array:
        y_train = np.load(os.path.join(self.root, 'y-train.npy'))
        answer = []
        for i in range(y_train.shape[1]):
            _, y_dist = torch.unique(torch.from_numpy(y_train[:, i]), return_counts=True)
            answer.append(y_dist.float())
        return answer

def _test():
    batch_size = 32
    data_dir = '/home/tom/github/comp-fair/db-seq/unimodal/depression/'
    
    # test `XYCTabDataset`
    xn_train = np.load(os.path.join(data_dir, 'xn-train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y-train.npy'))
    train_set = XYCTabDataset(xn_train, y_train)
    print('---------------------------------------- test1 ----------------------------------------')
    print('test `XYCTabDataset`')
    print(f'number of examples: {len(train_set)}')

    # test `XYCTabDataModule`
    data = XYCTabDataModule(data_dir, batch_size)
    train_loader = data.get_train_loader()
    val_loader = data.get_val_loader()
    test_loader = data.get_test_loader()
    x, y = next(iter(train_loader))
    print('\n---------------------------------------- test2 ----------------------------------------')
    print('test `XYCTabDataModule`')
    print('train dataloader')
    print(f'x.shape: {x.shape}, y.shape: {y.shape}')
    x, y = next(iter(val_loader))
    print('val dataloader')
    print(f'x.shape: {x.shape}, y.shape: {y.shape}')
    x, y = next(iter(test_loader))
    print('test dataloader')
    print(f'x.shape: {x.shape}, y.shape: {y.shape}')


if __name__ == '__main__':
    _test()
