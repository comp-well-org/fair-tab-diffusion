"""Utility functions for diffusion models.

Reference:
    - https://github.com/yandex-research/tab-ddpm/blob/main/tab_ddpm/utils.py
"""

import os
import math
import json
import torch
import numpy as np
import pandas as pd
import skops.io as sio
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from inspect import isfunction

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
        subset: float = 1.0,
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
        if subset < 1.0:
            n = int(subset * len(x))
            x = x.iloc[:n]
            y = y.iloc[:n]
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
    
    def get_feature_label_cols(self) -> tuple:
        feature = pd.read_csv(os.path.join(self.root, 'x_train.csv'), index_col=0)
        label = pd.read_csv(os.path.join(self.root, 'y_train.csv'), index_col=0)
        return feature.columns.tolist(), label.columns.tolist()

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: tensor of shape `[n,]`
        dim: embedding dimension
        max_period: maximum period
    
    Returns:
        sinusoidal timestep embeddings of shape `[n, dim]`
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half,
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, 'at least one argument must be a Tensor'

    # force variances to be tensors
    # broadcasting helps convert scalars to tensors, but it does not work for torch.exp()
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

def approx_standard_normal_cdf(x):
    a = 2.0
    b = 0.044715
    return 0.5 * (1.0 + torch.tanh(np.sqrt(a / np.pi) * (x + b * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    assert x.shape == means.shape == log_scales.shape
    in_max = 255.0
    clamp_min = 1e-12
    x_ref = 0.999
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / in_max)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / in_max)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=clamp_min))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=clamp_min))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -x_ref,
        log_cdf_plus,
        torch.where(x > x_ref, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=clamp_min))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def ohe_to_categories(ohe, k):
    k = torch.from_numpy(k)
    indices = torch.cat([torch.zeros((1,)), k.cumsum(dim=0)], dim=0).int().tolist()
    res = []
    for i in range(len(indices) - 1):
        res.append(ohe[:, indices[i]:indices[i + 1]].argmax(dim=1))
    return torch.stack(res, dim=1)

def log_1_min_a(a):
    offset = 1e-40
    return torch.log(1 - a.exp() + offset)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def exists(x):
    return x is not None

def extract(a, t, x_shape):
    t = t.to(a.device)
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i], num_classes[i]))
    x_onehot = torch.cat(onehots, dim=1)
    min_ref = 1e-30
    log_onehot = torch.log(x_onehot.float().clamp(min=min_ref))
    return log_onehot

def log_sum_exp_by_classes(x, slices):
    res = torch.zeros_like(x)
    for ixs in slices:
        res[:, ixs] = torch.logsumexp(x[:, ixs], dim=1, keepdim=True)
    assert x.size() == res.size()
    return res

@torch.jit.script
def log_sub_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m)) + m

@torch.jit.script
def sliced_logsumexp(x, slices):
    lse = torch.logcumsumexp(
        torch.nn.functional.pad(x, [1, 0, 0, 0], value=-float('inf')),
        dim=-1,
    )
    slice_starts = slices[:-1]
    slice_ends = slices[1:]
    slice_lse = log_sub_exp(lse[:, slice_ends], lse[:, slice_starts])
    slice_lse_repeated = torch.repeat_interleave(
        slice_lse,
        slice_ends - slice_starts, 
        dim=-1,
    )
    return slice_lse_repeated

def log_onehot_to_index(log_x):
    return log_x.argmax(1)
