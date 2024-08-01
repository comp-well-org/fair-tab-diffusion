import os
import time
import math
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Callable, List, Type, Union
from inspect import isfunction
import skops.io as sio
import sys

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

warnings.filterwarnings('ignore')

################################################################################
# data and utils
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
    
    def get_cat_label_mapping(self) -> tuple:
        cat_mapping = sio.load(os.path.join(self.root, 'cat_encoder.skops'))
        label_mapping = sio.load(os.path.join(self.root, 'label_encoder.skops'))
        return cat_mapping, label_mapping

    def get_data_description(self) -> dict:
        with open(os.path.join(self.root, 'desc.json'), 'r') as f:
            description = json.load(f)
        return description
    
    def inverse_transform(self, xn, y):
        # TODO: implement this
        # read desc, cat_mapping, label_mapping, fn, and then inverse transform xn and y
        pass
    
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

################################################################################
# model
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def reglu(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)

def geglu(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return reglu(x)

class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return geglu(x)

def _make_nn_module(module_type, *args) -> nn.Module:
    if module_type == 'ReGLU':
        return ReGLU()
    elif module_type == 'GEGLU':
        return GEGLU()
    elif isinstance(module_type, str):
        return getattr(nn, module_type)(*args)
    return module_type(*args)

class MLP(nn.Module):
    class Block(nn.Module):
        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: Union[str, Callable[[], nn.Module]],
        d_out: int,
    ) -> None:
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert activation not in {'ReGLU', 'GEGLU'}

        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ],
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
        cls: Type['MLP'],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
    ) -> 'MLP':
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            message = 'if d_layers contains more than two elements, then all elements except for the first and the last ones must be equal'
            assert len(set(d_layers[1:-1])) == 1, message
        return MLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x

class MLPDenoiseFn(nn.Module):
    def __init__(self, d_in, n_unq_y, is_y_cond, rtdl_params, dim_t=128):
        super().__init__()
        self.dim_t = dim_t
        self.n_unq_y = n_unq_y
        self.is_y_cond = is_y_cond

        rtdl_params['d_in'] = dim_t
        rtdl_params['d_out'] = d_in

        self.mlp = MLP.make_baseline(**rtdl_params)

        if self.n_unq_y > 0 and is_y_cond:
            self.label_emb = nn.Embedding(self.n_unq_y, dim_t)
        elif self.n_unq_y == 0 and is_y_cond:
            self.label_emb = nn.Linear(1, dim_t)
        
        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )

    def forward(self, x, timesteps, y=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if self.is_y_cond and y is not None:
            if self.n_unq_y > 0:
                y = y.squeeze()
            else:
                y = y.resize(y.size(0), 1).float()
            emb += F.silu(self.label_emb(y))
        x = self.proj(x) + emb
        return self.mlp(x)

################################################################################
# diffusion
def betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar: callable, max_beta: float = 0.2) -> np.ndarray:
    """Create a beta schedule that discretizes the given alpha_t_bar function.
    
    Defines the cumulative product of (1 - beta) over time from t = [0, 1].

    Args:
        num_diffusion_timesteps: number of diffusion steps
        alpha_bar: function that returns alpha_bar(t)
        max_beta: maximum beta value
    
    Returns:
        beta schedule of shape `(num_diffusion_timesteps,)`
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int, max_beta: float = 0.2) -> np.ndarray:
    """Get a pre-defined beta schedule for the given name.
    
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.Beta schedules may be added, but should not 
    be removed or changed once they are committed to maintain backwards compatibility.
    
    Raises:
        NotImplementedError: if schedule_name is not supported

    Args:
        schedule_name: name of the beta schedule
        num_diffusion_timesteps: number of diffusion steps
        max_beta: maximum beta value
    
    Returns:
        beta schedule of shape `(num_diffusion_timesteps,)`
    """
    if schedule_name == 'linear':
        # linear schedule from Ho et al, extended to work for any number of diffusion steps.
        start = 1e-4
        end = 0.02
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * start
        beta_end = scale * end
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64,
        )
    elif schedule_name == 'cosine':
        offset = 0.008
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + offset) / (1 + offset) * math.pi / 2) ** 2,
            max_beta=max_beta,
        )
    raise NotImplementedError(f'unknown beta schedule: {schedule_name}')

class GaussianMultinomialDiffusion(nn.Module):    
    def __init__(
        self,
        num_classes: np.array,
        num_numerical_features: int,
        denoise_fn: callable,
        num_timesteps: int = 1000,
        gaussian_loss_type: str = 'mse',
        gaussian_parametrization: str = 'eps',  
        multinomial_loss_type: str = 'vb_stochastic',
        parametrization: str = 'x0',
        scheduler: str = 'cosine',
        max_beta: float = 0.2,
        is_fair: bool = False,
        device: str = 'cpu',
    ):
        super().__init__()
        assert gaussian_parametrization in {'eps', 'x0'}
        assert multinomial_loss_type in ('vb_stochastic')
        assert parametrization in {'x0', 'direct'}

        # device
        device = torch.device(device)
        
        # fairness
        self.is_fair = is_fair
        
        # data
        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes  # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))]),
        ).to(device)

        # classes
        self.slices_for_classes = [np.arange(self.num_classes[0])]
        offsets = np.cumsum(self.num_classes)
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(np.append([0], offsets)).to(device)

        # diffusion
        self._denoise_fn = denoise_fn
        self._denoise_fn = self._denoise_fn.to(device)
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.scheduler = scheduler

        # intermediate variables
        alphas = 1. - get_named_beta_schedule(scheduler, num_timesteps, max_beta=max_beta)
        alphas = torch.tensor(alphas.astype('float64'))
        betas = 1. - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # gaussian diffusion
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.from_numpy(
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:])),
        ).float().to(device)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float().to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas.numpy()) / (1.0 - alphas_cumprod)
        ).float().to(device)

        # check that the diffusion is stable
        threshold = 1e-5
        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < threshold
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < threshold
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < threshold

        # convert to float32 and register buffers.
        self.register_buffer('alphas', alphas.float().to(device))
        self.register_buffer('log_alpha', log_alpha.float().to(device))
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float().to(device))
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float().to(device))
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float().to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.float().to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float().to(device))
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next.float().to(device))
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod.float().to(device))
        self.register_buffer('lt_history', torch.zeros(num_timesteps))
        self.register_buffer('lt_count', torch.zeros(num_timesteps))
        
    # gaussian part
    def gaussian_q_mean_variance(self, x_start, t):
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_1_min_cumprod_alpha, t, x_start.shape,
        )
        return mean, variance, log_variance
    
    def gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        answer = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        answer += extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        return answer
    
    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape        
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape,
        )
        b1 = posterior_mean.shape[0]
        b2 = posterior_variance.shape[0]
        b3 = posterior_log_variance_clipped.shape[0]
        b4 = x_start.shape[0]
        assert b1 == b2 == b3 == b4
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def gaussian_p_mean_variance(self, model_output, x_t, t):
        b, _ = x_t.shape[:2]
        assert t.shape == (b,)

        model_variance = torch.cat([self.posterior_variance[1].unsqueeze(0).to(x_t.device), (1. - self.alphas)[1:]], dim=0)
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, t, x_t.shape)
        model_log_variance = extract(model_log_variance, t, x_t.shape)

        if self.gaussian_parametrization == 'eps':
            pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
        elif self.gaussian_parametrization == 'x0':
            pred_xstart = model_output
        else:
            raise NotImplementedError
        
        # print('pred_xstart std', torch.std(pred_xstart))
        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t,
        )
        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape)

        return {
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': model_log_variance,
            'pred_xstart': pred_xstart,
        }
    
    def _vb_terms_bpd(self, model_output, x_start, x_t, t):
        true_mean, _, true_log_variance_clipped = self.gaussian_q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t,
        )
        out = self.gaussian_p_mean_variance(model_output, x_t, t)
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out['mean'], out['log_variance'],
        )
        kl = mean_flat(kl) / np.log(2)
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out['mean'], log_scales=0.5 * out['log_variance'],
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2)

        # at the first timestep return the decoder NLL
        # otherwise return $KL(q(x_{t - 1} | x_t, x_0) || p(x_{t - 1} | x_t))$
        output = torch.where((t == 0), decoder_nll, kl)
        out_mean = out['mean'] 
        return {
            'output': output, 
            'pred_xstart': out['pred_xstart'], 
            'out_mean': out_mean, 
            'true_mean': true_mean,
        }

    def _prior_gaussian(self, x_start: torch.Tensor):
        """Get the prior KL term for the variational lower-bound, measured in bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.
        
        Args:
            x_start: tensor of shape `[N, C, ...]`
        
        Returns:
            KL values in bits of shape `[N,]`
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.gaussian_q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0., logvar2=0.,
        )
        return mean_flat(kl_prior) / np.log(2)
    
    def _gaussian_loss(self, model_out, x_start, x_t, t, noise):
        if self.gaussian_loss_type == 'mse':
            loss = mean_flat((noise - model_out) ** 2)
        elif self.gaussian_loss_type == 'kl':
            loss = self._vb_terms_bpd(
                model_output=model_out,
                x_start=x_start,
                x_t=x_t,
                t=t,
            )['output']
        return loss

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        answer = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        # print('coef', extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)[0, 0])
        # print('`x_t` std', torch.std(x_t), 'answer std', torch.std(answer))
        return answer
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        answer = (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return answer
    
    def gaussian_p_sample(
        self,
        model_out,
        x_t,
        t,
    ):
        # print('model_out_num std', torch.std(model_out))
        out = self.gaussian_p_mean_variance(
            model_out,
            x_t,
            t,
        )
        # no noise when t == 0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        sample = out['mean'] + nonzero_mask * torch.exp(0.5 * out['log_variance']) * noise
        pred_xstart = out['pred_xstart']
        return {
            'sample': sample, 
            'pred_xstart': pred_xstart,
        }
    
    # multinomial part
    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl
    
    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)
        # $alpha_t * E[xt] + (1 - alpha_t) 1 / K$
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(self.num_classes_expanded),
        )
        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)
        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded),
        )
        return log_probs
    
    def predict_start(self, model_out, log_x_t):
        assert model_out.size(0) == log_x_t.size(0)
        assert model_out.size(1) == self.num_classes.sum(), f'{model_out.size()}'
        log_pred = torch.empty_like(model_out)
        for ix in self.slices_for_classes:
            log_pred[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return log_pred
    
    def q_posterior(self, log_x_start, log_x_t, t):
        t_minus_1 = t - 1
        # remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_ev_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)
        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.to(log_x_start.device).view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_ev_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_ev_qxtmin_x0.to(torch.float32))
        unnormed_logprobs = log_ev_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)
        log_ev_xtmin_given_xt_given_xstart = unnormed_logprobs - sliced_logsumexp(unnormed_logprobs, self.offsets)
        return log_ev_xtmin_given_xt_given_xstart
    
    def p_pred(self, model_out, log_x, t):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(model_out, log_x)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t,
            )
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(model_out, log_x)
        else:
            raise ValueError
        return log_model_pred
    
    @torch.no_grad()
    def p_sample(self, model_out, log_x, t):
        model_log_prob = self.p_pred(model_out, log_x=log_x, t=t)
        out = self.log_sample_categorical(model_log_prob)
        return out
    
    def log_sample_categorical(self, logits):
        full_sample = []
        offset = 1e-30
        for i in range(len(self.num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + offset) + offset)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = index_to_log_onehot(full_sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_ev_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_ev_qxt_x0)
        return log_sample
    
    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxt_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes_expanded * torch.ones_like(log_qxt_prob))

        kl_prior = self.multinomial_kl(log_qxt_prob, log_half_prob)
        return sum_except_batch(kl_prior)
    
    def compute_lt(self, model_out, log_x_start, log_x_t, t, detach_mean=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t,
        )
        log_model_prob = self.p_pred(model_out, log_x=log_x_t, t=t)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def _multinomial_loss(self, model_out, log_x_start, log_x_t, t, pt):
        if self.multinomial_loss_type == 'vb_stochastic':
            kl = self.compute_lt(model_out, log_x_start, log_x_t, t)
            kl_prior = self.kl_prior(log_x_start)
            # upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior
            return vb_loss

        raise ValueError()

    def sample_time(self, b, device, method='uniform'):
        offset = 1e-10
        bias = 0.0001
        if method == 'importance':
            if not (self.lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')
            lt_sqrt = torch.sqrt(self.lt_history + offset) + bias
            lt_sqrt[0] = lt_sqrt[1]
            pt_all = (lt_sqrt / lt_sqrt.sum()).to(device)
            t = torch.multinomial(pt_all, n_samples=b, replacement=True).to(device)
            pt = pt_all.gather(dim=0, index=t)
            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        
        raise ValueError

    def mixed_loss(self, x, cond):
        b = x.shape[0]
        device = x.device
        t, pt = self.sample_time(b, device, 'uniform')

        # split numerical and categorical features
        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:]
        
        # process numerical and categorical features separately
        x_num_t = x_num
        log_x_cat_t = x_cat
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)
        if x_cat.shape[1] > 0:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)
        x_t = torch.cat([x_num_t, log_x_cat_t], dim=1)

        # denoise
        model_out = self._denoise_fn(x_t, t, cond)
        model_out_num = model_out[:, :self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features:]

        # compute losses
        loss_multi = torch.zeros((1,)).float()
        loss_gauss = torch.zeros((1,)).float()
        if x_cat.shape[1] > 0:
            loss_multi = self._multinomial_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt) / len(self.num_classes)
        if x_num.shape[1] > 0:
            loss_gauss = self._gaussian_loss(model_out_num, x_num, x_num_t, t, noise)
        return loss_multi.mean(), loss_gauss.mean()
    
    def sample(self, batch_size, c_dist: List[torch.Tensor], num_generated: int):
        with torch.no_grad():
            b = batch_size
            device = self.log_alpha.device
            z_norm = torch.randn((b, self.num_numerical_features), device=device)

            has_cat = self.num_classes[0] != 0
            log_z = torch.zeros((b, 0), device=device).float()
            if has_cat:
                uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
                log_z = self.log_sample_categorical(uniform_logits)

            columns = []
            for dist in c_dist:
                column = torch.multinomial(
                    dist,
                    num_samples=batch_size,
                    replacement=True,
                )
                columns.append(column)
            matrix = torch.stack(columns, dim=1)
            cond = matrix.long().to(device)
        
            if self.is_fair:
                instruction = cond
            else:
                instruction = cond[:, 0].unsqueeze(1)  # only use the first column which is the outcome
        
        x_t = torch.cat([z_norm, log_z], dim=1).float()
        for i in reversed(range(0, self.num_timesteps)):
            with torch.no_grad():
                if i != self.num_timesteps - 1:
                    print(f'sampling timestep {self.num_timesteps - 1:04d} -> {i:04d} -- generated: {num_generated:04d}', end='\r')
                else:
                    print(f'sampling timestep {self.num_timesteps - 1:04d} -> {i:04d} -- generated: {num_generated:04d}')
                t = torch.full((b,), i, device=device, dtype=torch.long)
                model_out = self._denoise_fn(
                    x_t,
                    t,
                    instruction,
                )
                model_out_num = model_out[:, :self.num_numerical_features]
                model_out_cat = model_out[:, self.num_numerical_features:]
                z_norm = self.gaussian_p_sample(model_out_num, z_norm, t)['sample']
                if has_cat:
                    log_z = self.p_sample(model_out_cat, log_z, t)
            x_t = torch.cat([z_norm, log_z], dim=1).float()
    
        with torch.no_grad():
            z_ohe = torch.exp(log_z).round()
            z_cat = log_z
            if has_cat:
                z_cat = ohe_to_categories(z_ohe, self.num_classes)
            sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, cond.cpu()

    def sample_all(self, n_samples: int, c_dist: List[torch.Tensor], batch_size: int = 1000):
        sample_fn = self.sample
        
        all_samples = []
        all_cond = []
        num_generated = 0
        while num_generated < n_samples:
            samples, cond = sample_fn(batch_size, c_dist, num_generated)
            mask_nam = torch.any(samples.isnan(), dim=1)
            samples = samples[~mask_nam]
            cond = cond[~mask_nam]
            all_samples.append(samples)
            all_cond.append(cond)
            if samples.shape[0] != batch_size:
                raise ValueError('found nan during sampling')
            num_generated += samples.shape[0]
        
        x_gen = torch.cat(all_samples, dim=0)[:n_samples]
        cond_gen = torch.cat(all_cond, dim=0)[:n_samples]
        
        return x_gen, cond_gen

################################################################################
# train and sample
class XYCTabTrainer:
    def __init__(
        self,
        n_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_non_improve: int = 10,
        is_fair: bool = False,
        device: str = 'cpu',
    ) -> None:
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_non_improve = max_non_improve
        self.is_fair = is_fair
        self.device = torch.device(device)
    
    def updata_args(
        self, 
        n_epochs: int = None, 
        lr: float = None, 
        weight_decay: float = None, 
        max_non_improve: int = None,
        is_fair: bool = None,
        device: str = None,
    ) -> None:
        if n_epochs is not None:
            self.n_epochs = n_epochs
        if lr is not None:
            self.lr = lr
        if weight_decay is not None:
            self.weight_decay = weight_decay
        if max_non_improve is not None:
            self.max_non_improve = max_non_improve
        if is_fair is not None:
            self.is_fair = is_fair
        if device is not None:
            self.device = torch.device(device)

    def prepare_model(self, model: GaussianMultinomialDiffusion):
        self.model = model
        self.model.to(self.device)

    def prepare_data(self, data: XYCTabDataModule, normalize: bool = True):
        self.data = data
        self.train_loader = data.get_dataloader('train', normalize=normalize)
        self.eval_loader = data.get_dataloader('eval', normalize=normalize)
        self.test_loader = data.get_dataloader('test', normalize=normalize)
        
    def fit(self, model: GaussianMultinomialDiffusion, data: XYCTabDataModule, exp_dir: str = None):
        if exp_dir is not None:
            os.makedirs(exp_dir, exist_ok=True)
        
        # prepare
        self.prepare_model(model)
        self.prepare_data(data)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay,
        )
        
        # record
        self.state = {'epoch': 0}
        self.loss_record = {'mloss': np.inf, 'gloss': np.inf, 'tloss': np.inf, 'keeps': 0, 'break': False, 'epoch': 0}
        self.epoch_loss_history = pd.DataFrame(columns=['epoch', 'mloss', 'gloss', 'tloss'])
        
        # train
        t_loss_min = np.inf
        for epoch in range(self.n_epochs):
            self.state['epoch'] = epoch + 1
            mloss, gloss, tloss = self._fit_epoch(self.model, self.train_loader)
            self.loss_record['keeps'] += 1
            if tloss < t_loss_min:
                self.loss_record['mloss'] = mloss
                self.loss_record['gloss'] = gloss
                self.loss_record['tloss'] = tloss
                self.loss_record['keeps'] = 0
                self.loss_record['epoch'] = epoch + 1
                t_loss_min = tloss
                if exp_dir is not None:
                    self.save_model(os.path.join(exp_dir, 'diffusion.pt'))
            if self.loss_record['break']:
                break
            curr_idx = len(self.epoch_loss_history)
            self.epoch_loss_history.loc[curr_idx] = [epoch + 1, mloss, gloss, tloss]
            self.epoch_loss_history.to_csv(os.path.join(exp_dir, 'loss.csv'), index=False)
            self._anneal_lr(epoch)
        
        print()
        print('training complete ^_^')
    
    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
    
    def _fit_epoch(self, model: callable, data_loader: torch.utils.data.DataLoader) -> tuple:
        total_mloss = 0
        total_gloss = 0
        curr_count = 0
        for batch_idx, (x, y) in enumerate(data_loader):
            self.optimizer.zero_grad()
            if self.is_fair:
                x, y = x.to(self.device), y.long().to(self.device)
                loss_multi, loss_gauss = model.mixed_loss(x, y)
            else:
                y = y[:, 0].unsqueeze(1)  # only use the first column which is the label
                x, y = x.to(self.device), y.long().to(self.device)
                loss_multi, loss_gauss = model.mixed_loss(x, y)
            loss = loss_multi + loss_gauss
            loss.backward()
            self.optimizer.step()
            
            # record loss
            with torch.no_grad():
                total_mloss += loss_multi.item() * x.shape[0]
                total_gloss += loss_gauss.item() * x.shape[0]
                curr_count += x.shape[0]
                mloss = np.around(total_mloss / curr_count, 4)
                gloss = np.around(total_gloss / curr_count, 4)
                tloss = np.around(mloss + gloss, 4)
            
            keeps = self.loss_record['keeps']
            curr_epoch = self.state['epoch']
            upper_limit = self.max_non_improve
            if keeps > upper_limit:
                msg = f'best results so far -> epoch: {curr_epoch:04}/{self.n_epochs:04}, mloss: {mloss:.4f}, gloss: {gloss:.4f}, tloss: {tloss:.4f} -- best: {self.loss_record["tloss"]:.4f}'
                print(msg)
                self.loss_record['break'] = True
                print(f'the model has not improved for {self.max_non_improve} epochs, stopping training')
                break
            
            if batch_idx == len(data_loader) - 1 and curr_epoch == self.n_epochs:
                msg = f'best results so far -> epoch: {curr_epoch:04}/{self.n_epochs:04}, mloss: {mloss:.4f}, gloss: {gloss:.4f}, tloss: {tloss:.4f} -- best: {self.loss_record["tloss"]:.4f}'
                print(msg)
            else:
                msg = f'best results so far -> epoch: {curr_epoch:04}/{self.n_epochs:04}, mloss: {mloss:.4f}, gloss: {gloss:.4f}, tloss: {tloss:.4f} -- best: {self.loss_record["tloss"]:.4f}'
                print(msg, end='\r')
        return mloss, gloss, tloss
        
    def _anneal_lr(self, epoch):
        frac_done = epoch / self.n_epochs
        lr = self.lr * (1 - frac_done / 2)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

################################################################################
# main
def main():
    # global variables
    device = torch.device('cuda:1')
    
    # TODO: configs
    data_config = {
        'path': '/rdf/db/public-tabular-datasets/',
        'name': 'adult',
        'batch_size': 128,
    }
    rtdl_params = {
        'd_layers': [512, 512],
        'dropout': 0.0,
    }
    dim_t = 128
    train_config = {
        'n_epochs': 2,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'max_non_improve': 10,
        'is_fair': False,
    }
    
    # data
    data_module = XYCTabDataModule(
        root=os.path.join(data_config['path'], data_config['name']),
        batch_size=data_config['batch_size'],
    )
    data_desc = data_module.get_data_description()
    d_in = data_desc['d_oh_x']
    n_unq_y = data_desc['n_unq_y']
    n_unq_cat_od_x_lst = np.array(data_desc['n_unq_cat_od_x_lst'])
    d_num_x = data_desc['d_num_x']
    
    # model
    denoise_fn = MLPDenoiseFn(
        d_in=d_in,
        n_unq_y=n_unq_y,
        is_y_cond=True,
        rtdl_params=rtdl_params,
        dim_t=dim_t,
    )
    
    # diffusion
    diffusion = GaussianMultinomialDiffusion(
        num_classes=n_unq_cat_od_x_lst,
        num_numerical_features=d_num_x,
        denoise_fn=denoise_fn,
        device='cuda:1',
        scheduler='cosine',
        max_beta=0.2,
        num_timesteps=1000,
        is_fair=train_config['is_fair'],
        gaussian_parametrization='eps',
    )
    
    # training
    trainer = XYCTabTrainer(
        n_epochs=train_config['n_epochs'],
        lr=train_config['lr'],
        weight_decay=train_config['weight_decay'],
        max_non_improve=train_config['max_non_improve'],
        is_fair=train_config['is_fair'],
        device=device,
    )
    start_time = time.time()
    trainer.fit(diffusion, data_module, 'ckpt')
    end_time = time.time()
    print(f'training time: {end_time - start_time:.2f} seconds')
    
    # sampling
    print()
    c_dist = data_module.get_empirical_dist()
    xn, cond = diffusion.sample_all(1000, [c_dist[0]], batch_size=500)
    print(f'synthetic data shape: {list(xn.shape)}, synthetic cond shape: {list(cond.shape)}')

if __name__ == '__main__':
    main()
