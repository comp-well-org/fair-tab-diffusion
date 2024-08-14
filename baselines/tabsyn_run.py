import os
import sys
import math
import time
import json
import torch
import warnings
import argparse
import torch.optim
import numpy as np
import pandas as pd
import skops.io as sio
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

# importing the required files from the parent directory
from lib import load_config, copy_file, load_json
from src.evaluate.metrics import evaluate_syn_data, print_metric
from constant import DB_PATH, EXPS_PATH

warnings.filterwarnings('ignore')

################################################################################
# data
class TabularDataset(Dataset):
    def __init__(self, x_num, x_cat):
        if isinstance(x_num, np.ndarray):
            x_num = torch.from_numpy(x_num).float()
        if isinstance(x_cat, np.ndarray):
            x_cat = torch.from_numpy(x_cat).long()
        self.x_num = x_num
        self.x_cat = x_cat

    def __getitem__(self, idx):
        this_num = self.x_num[idx]
        this_cat = self.x_cat[idx]
        sample = (this_num, this_cat)
        return sample

    def __len__(self):
        return self.x_num.shape[0]

def preprocess(data_dir):
    xn_train = pd.read_csv(os.path.join(data_dir, 'xn_train.csv'), index_col=0)
    xn_eval = pd.read_csv(os.path.join(data_dir, 'xn_eval.csv'), index_col=0)
    xn_test = pd.read_csv(os.path.join(data_dir, 'xn_test.csv'), index_col=0)
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), index_col=0)
    y_eval = pd.read_csv(os.path.join(data_dir, 'y_eval.csv'), index_col=0)
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), index_col=0)
    
    # only the first column of y_train, y_eval, y_test is used
    y_train = y_train.iloc[:, 0]
    y_eval = y_eval.iloc[:, 0]
    y_test = y_test.iloc[:, 0]
    
    data_train = pd.concat([xn_train, y_train], axis=1)
    data_eval = pd.concat([xn_eval, y_eval], axis=1)
    data_test = pd.concat([xn_test, y_test], axis=1)
    # print(data_train.head())
    
    with open(os.path.join(data_dir, 'desc.json')) as f:
        desc = json.load(f)
    # print(json.dumps(desc, indent=4))
    
    categories = desc['n_unq_cat_od_x_lst'] + [desc['n_unq_y']]
    d_numerical = desc['d_num_x']
    X_train_num = data_train.iloc[:, :d_numerical].values
    X_eval_num = data_eval.iloc[:, :d_numerical].values
    X_test_num = data_test.iloc[:, :d_numerical].values
    X_train_cat = data_train.iloc[:, d_numerical:].values
    X_eval_cat = data_eval.iloc[:, d_numerical:].values
    X_test_cat = data_test.iloc[:, d_numerical:].values
    X_num_sets = (X_train_num, X_eval_num, X_test_num)
    X_cat_sets = (X_train_cat, X_eval_cat, X_test_cat)
    
    return X_num_sets, X_cat_sets, categories, d_numerical

################################################################################
# model
class Tokenizer(nn.Module):
    def __init__(self, d_numerical, categories, d_token, bias):
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self):
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num, x_cat):
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ],
            )
            x = x + bias[None]
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout, initialization='kaiming'):

        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head).transpose(1, 2).reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv, key_compression=None, value_compression=None):
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        a = q @ k.transpose(1, 2)
        b = math.sqrt(d_head_key)
        attention = F.softmax(a / b, dim=-1)

        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value).transpose(1, 2).reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x
        
class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_out: int,
        d_ffn_factor: int,
        attention_dropout=0.0,
        ffn_dropout=0.0,
        residual_dropout=0.0,
        activation='relu',
        prenormalization=True,
        initialization='kaiming',      
    ):
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization,
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden,
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                },
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()

            self.layers.append(layer)
        self.activation = nn.ReLU()
        self.last_activation = nn.ReLU()
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                x_residual,
                x_residual,
            )
            x = self._end_residual(x, x_residual, layer, 0)
            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        return x

class AE(nn.Module):
    def __init__(self, hid_dim, n_head):
        super(AE, self).__init__()
        self.hid_dim = hid_dim
        self.n_head = n_head
        self.encoder = MultiheadAttention(hid_dim, n_head)
        self.decoder = MultiheadAttention(hid_dim, n_head)

    def get_embedding(self, x):
        return self.encoder(x, x).detach() 

    def forward(self, x):
        z = self.encoder(x, x)
        h = self.decoder(z, z)
        return h

class VAE(nn.Module):
    def __init__(self, d_numerical, categories, num_layers, hid_dim, n_head=1, factor=4, bias=True):
        super(VAE, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.hid_dim = hid_dim
        d_token = hid_dim
        self.n_head = n_head
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias=bias)
        self.encoder_mu = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)
        self.encoder_logvar = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)
        self.decoder = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

    def get_embedding(self, x):
        return self.encoder_mu(x, x).detach() 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_num, x_cat):
        x = self.Tokenizer(x_num, x_cat)

        mu_z = self.encoder_mu(x)
        std_z = self.encoder_logvar(x)

        z = self.reparameterize(mu_z, std_z)
        h = self.decoder(z[:, 1:])
        
        return h, mu_z, std_z

class Reconstructor(nn.Module):
    def __init__(self, d_numerical, categories, d_token):
        super(Reconstructor, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token
        
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))  
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList()

        for d in categories:
            recon = nn.Linear(d_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h):
        h_num = h[:, :self.d_numerical]
        h_cat = h[:, self.d_numerical:]

        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        recon_x_cat = []

        for i, recon in enumerate(self.cat_recons):
            recon_x_cat.append(recon(h_cat[:, i]))

        return recon_x_num, recon_x_cat

class ModelVAE(nn.Module):
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head=1, factor=4, bias=True):
        super(ModelVAE, self).__init__()

        self.VAE = VAE(d_numerical, categories, num_layers, d_token, n_head=n_head, factor=factor, bias=bias)
        self.Reconstructor = Reconstructor(d_numerical, categories, d_token)

    def get_embedding(self, x_num, x_cat):
        x = self.Tokenizer(x_num, x_cat)
        return self.VAE.get_embedding(x)

    def forward(self, x_num, x_cat):
        h, mu_z, std_z = self.VAE(x_num, x_cat)
        recon_x_num, recon_x_cat = self.Reconstructor(h)
        return recon_x_num, recon_x_cat, mu_z, std_z

class EncoderModel(nn.Module):
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head, factor, bias=True):
        super(EncoderModel, self).__init__()
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias)
        self.VAE_Encoder = Transformer(num_layers, d_token, n_head, d_token, factor)

    def load_weights(self, pretrained_vae):
        self.Tokenizer.load_state_dict(pretrained_vae.VAE.Tokenizer.state_dict())
        self.VAE_Encoder.load_state_dict(pretrained_vae.VAE.encoder_mu.state_dict())

    def forward(self, x_num, x_cat):
        x = self.Tokenizer(x_num, x_cat)
        z = self.VAE_Encoder(x)
        return z

class DecoderModel(nn.Module):
    def __init__(self, num_layers, d_numerical, categories, d_token, n_head, factor, bias=True):
        super(DecoderModel, self).__init__()
        self.VAE_Decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.Detokenizer = Reconstructor(d_numerical, categories, d_token)
        
    def load_weights(self, pretrained_vae):
        self.VAE_Decoder.load_state_dict(pretrained_vae.VAE.decoder.state_dict())
        self.Detokenizer.load_state_dict(pretrained_vae.Reconstructor.state_dict())

    def forward(self, z):
        h = self.VAE_Decoder(z)
        x_hat_num, x_hat_cat = self.Detokenizer(h)
        return x_hat_num, x_hat_cat

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

def reglu(x: Tensor) -> Tensor:
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)

def geglu(x: Tensor) -> Tensor:
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)

class GEGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=512):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )
    
    def forward(self, x, noise_labels, class_labels=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
    
        x = self.proj(x) + emb
        return self.mlp(x)

class Precond(nn.Module):
    def __init__(
        self,
        denoise_fn,
        hid_dim,
        sigma_min=0,                
        sigma_max=float('inf'),     
        sigma_data=0.5,             
    ):
        super().__init__()
        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma):
        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

        assert F_x.dtype == dtype
        d_x = c_skip * x + c_out * F_x.to(torch.float32)
        return d_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
class Model(nn.Module):
    def __init__(self, denoise_fn, hid_dim, p_mean=-1.2, p_std=1.2, sigma_data=0.5, gamma=5, opts=None, pfgmpp=False):
        super().__init__()
        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(p_mean, p_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)

    def forward(self, x):
        loss = self.loss_fn(self.denoise_fn_D, x)
        return loss.mean(-1).mean()

################################################################################
# diffusion utils
SIGMA_MIN = 0.002
SIGMA_MAX = 80
RHO = 7
S_CHURN = 1
S_MIN = 0
S_MAX = float('inf')
S_NOISE = 1

def sample(net, num_samples, dim, num_steps=50, device='cuda:0'):
    latents = torch.randn([num_samples, dim], device=device) * 1000

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / RHO) + step_indices / (num_steps - 1) * (sigma_min ** (1 / RHO) - sigma_max ** (1 / RHO))) ** RHO
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next)

    return x_next

def sample_step(net, num_steps, i, t_cur, t_next, x_next):
    x_cur = x_next
    
    # increase noise temporarily
    gamma = min(S_CHURN / num_steps, np.sqrt(2) - 1) if S_MIN <= t_cur <= S_MAX else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur) 
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_NOISE * torch.randn_like(x_cur)
    
    # euler step
    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # apply 2nd order correction
    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, denosie_fn, data, labels, augment_pipe=None):
        rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
        n = torch.randn_like(y) * sigma
        D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, d=128, n=3072, opts=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = d
        self.N = n

    def __call__(self, denosie_fn, data, labels=None, augment_pipe=None, stf=False, pfgmpp=False, ref_data=None):
        if pfgmpp:
            rnd_uniform = torch.rand(data.shape[0], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

            r = sigma.double() * np.sqrt(self.D).astype(np.float64)
            # sampling form inverse-beta distribution
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2., size=data.shape[0]).astype(np.double)

            samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(data.device).double()
            # sampling from p_r(R) by change-of-variable
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            
            # uniformly sample the angle direction
            gaussian = torch.randn(data.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            
            # construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = perturbation_x.view_as(y)
            D_yn = denosie_fn(y + n, sigma, labels,  augment_labels=augment_labels)
        else:
            rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = torch.randn_like(y) * sigma
            D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)
        return loss

class EDMLoss:
    def __init__(self, p_mean=-1.2, p_std=1.2, sigma_data=0.5, hid_dim=100, gamma=5, opts=None):
        self.P_mean = p_mean
        self.P_std = p_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts

    def __call__(self, denoise_fn, data):
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = denoise_fn(y + n, sigma)
    
        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)
        return loss

################################################################################
# latent utils
@torch.no_grad()
def split_num_cat_target(syn_data, categories, d_numerical, pre_decoder, token_dim):
    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)
    syn_data = torch.tensor(syn_data)
    pre_decoder = pre_decoder.to(syn_data.device)
    norm_input = pre_decoder(syn_data)
    x_hat_num, x_hat_cat = norm_input

    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim=-1))

    syn_num = x_hat_num.cpu().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()

    return syn_num, syn_cat

################################################################################
# training
def compute_loss(x_num, x_cat, recon_x_num, recon_x_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (x_num - recon_x_num).pow(2).mean()
    ce_loss = 0

    for _, x_cat in enumerate(recon_x_cat):
        if x_cat is not None:
            x_hat = x_cat.argmax(dim=-1)
            ce_loss += ce_loss_fn(x_cat, x_hat)
    
    ce_loss /= len(recon_x_cat)
    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld

def train_latent_model(
    model, pre_encoder, pre_decoder, optimizer, scheduler, num_epochs,
    train_loader, x_train_num, x_train_cat, x_eval_num, x_eval_cat,
    model_save_path, encoder_save_path, decoder_save_path, ckpt_dir,
    min_beta, max_beta, lambd, 
    device, 
):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    best_train_loss = float('inf')

    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    beta = max_beta
    
    for epoch in range(num_epochs):
        if epoch == num_epochs - 1:
            print(f'training latent -> epoch: {epoch + 1}/{num_epochs}')
        else:
            print(f'training latent -> epoch: {epoch + 1}/{num_epochs}', end='\r')
            
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0

        curr_count = 0

        for _, (batch_num, batch_cat) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)

            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)
        
            loss_mse, loss_ce, loss_kld = compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)

            loss = loss_mse + loss_ce + beta * loss_kld
            loss.backward()
            optimizer.step()

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl += loss_kld.item() * batch_length
        
        model.eval()
        with torch.no_grad():
            x_eval_num = x_eval_num.to(device)
            x_eval_cat = x_eval_cat.to(device)
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(x_eval_num, x_eval_cat)

            val_mse_loss, val_ce_loss, val_kl_loss = compute_loss(x_eval_num, x_eval_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
            val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()    

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != current_lr:
                current_lr = new_lr
                
            train_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd
    
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        X_train_num = x_train_num.to(device)
        X_train_cat = x_train_cat.to(device)
        
        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()
        np.save(f'{ckpt_dir}/train_z.npy', train_z)

def get_input_train(ckpt_dataset_dir):
    embedding_save_path = f'{ckpt_dataset_dir}/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)
    return train_z

def train_diffusion_model(
    model, optimizer, scheduler, num_epochs, train_loader, ckpt_dir, device,
):
    model.train()

    best_loss = float('inf')
    curr_loss = 0
    patience = 0
    for epoch in range(num_epochs):
        if epoch == num_epochs - 1:
            print(f'training diffusion -> epoch: {epoch + 1}/{num_epochs}, loss: {curr_loss:.4f} -- best: {best_loss:.4f}')
        else:
            print(f'training diffusion -> epoch: {epoch + 1}/{num_epochs}, loss: {curr_loss:.4f} -- best: {best_loss:.4f}', end='\r')

        batch_loss = 0.0
        len_input = 0
        for batch in train_loader:
            inputs = batch.float().to(device)
            loss = model(inputs)
        
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f'{ckpt_dir}/diffusion.pt')
        else:
            patience += 1
            if patience == 500:
                print('early stopping')
                break

################################################################################
# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--exp_name', type=str, default='check')
    parser.add_argument('--train', action='store_true', help='training', default=True)
    parser.add_argument('--sample', action='store_true', help='sampling', default=True)
    parser.add_argument('--eval', action='store_true', help='evaluation', default=True)
    
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError('config file is required')

    # global variables
    WD = 0
    D_TOKEN = 4

    N_HEAD = 1
    FACTOR = 32
    NUM_LAYERS = 2
    
    # configs
    exp_config = config['exp']
    data_config = config['data']
    train_config = config['train']
    sample_config = config['sample']
    eval_config = config['eval']
    
    device = exp_config['device']
    seed = exp_config['seed']
    batch_size = data_config['batch_size']
    n_epochs = train_config['n_epochs']
    lr = train_config['lr'] 
    n_seeds = sample_config['n_seeds']
    
    # experimental directory
    exp_dir = os.path.join(
        exp_config['home'], 
        data_config['name'],
        exp_config['method'],
        args.exp_name,
    )
    copy_file(
        os.path.join(exp_dir), 
        args.config,
    )
    
    # message
    print(f'config file: {args.config}')
    print('-' * 80)

    # data
    dataset_dir = os.path.join(data_config['path'], data_config['name'])
    data_desc = load_json(os.path.join(dataset_dir, 'desc.json'))
    ckpt_dir = os.path.join(exp_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    norm_fn = sio.load(os.path.join(dataset_dir, 'fn.skops'))
    feature_cols = data_desc['col_names']
    label_cols = [data_desc['label_col_name']]
    X_num_sets, X_cat_sets, categories, d_numerical = preprocess(dataset_dir)
    X_train_num, X_eval_num, X_test_num = X_num_sets
    X_train_cat, X_eval_cat, X_test_cat = X_cat_sets
    X_train_num, X_eval_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_eval_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_eval_cat, X_test_cat = torch.tensor(X_train_cat).long(), torch.tensor(X_eval_cat).long(), torch.tensor(X_test_cat).long()
    
    train_data = TabularDataset(X_train_num, X_train_cat)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    # model
    model = ModelVAE(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR, bias=True)
    model = model.to(device)
    num_params_vae = sum(p.numel() for p in model.parameters())

    pre_encoder = EncoderModel(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR).to(device)
    pre_decoder = DecoderModel(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR).to(device)
    pre_encoder.eval()
    pre_decoder.eval()
    num_params_encoder = sum(p.numel() for p in pre_encoder.parameters())
    num_params_decoder = sum(p.numel() for p in pre_decoder.parameters())
    
    num_params = num_params_vae + num_params_encoder + num_params_decoder
    with open(os.path.join(exp_dir, 'params.txt'), 'w') as f:
        f.write(f'number of parameters: {num_params}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)
    
    # training latent model
    model_save_path = f'{ckpt_dir}/model.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'
    
    start_time = time.time()
    train_latent_model(
        model, pre_encoder, pre_decoder, optimizer, scheduler, 10000,
        train_loader, X_train_num, X_train_cat, X_eval_num, X_eval_cat,
        model_save_path, encoder_save_path, decoder_save_path, ckpt_dir,
        min_beta=0.1, max_beta=1.0, lambd=0.95,
        device=device,
    )
    end_time = time.time()
    with open(os.path.join(exp_dir, 'time.txt'), 'w') as f:
        time_msg = f'training time (latent): {end_time - start_time:.2f} seconds with {10000} epochs'
        f.write(time_msg)
    
    # training diffusion model
    train_z = get_input_train(ckpt_dir)
    in_dim = train_z.shape[1] 
    mean = train_z.mean(0)
    train_z = (train_z - mean) / 2
    train_data = train_z
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print('the number of parameters:', num_params)
    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)
    
    if args.train:
        # train
        start_time = time.time()
        train_diffusion_model(
            model, optimizer, scheduler, n_epochs, train_loader, ckpt_dir, device,
        )
        end_time = time.time()
        with open(os.path.join(exp_dir, 'time.txt'), 'a') as f:
            time_msg = f'\ntraining time (diffusion): {end_time - start_time:.2f} seconds with {n_epochs} epochs'
            f.write(time_msg)
        print()
        
    if args.sample:
        # loading    
        in_dim = train_z.shape[1] 
        mean = train_z.mean(0)
        denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
        
        model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)
        model.load_state_dict(torch.load(f'{ckpt_dir}/diffusion.pt'))
        
        # sampling
        start_time = time.time()
        for i in range(n_seeds):
            random_seed = seed + i
            random_seed = seed + i
            torch.manual_seed(random_seed)
            
            num_samples = train_z.shape[0]
            sample_dim = in_dim

            x_next = sample(model.denoise_fn_D, num_samples, sample_dim, device=device)
            x_next = x_next * 2 + mean.to(device)

            syn_data = x_next.float().cpu().numpy()
            
            embedding_save_path = f'{ckpt_dir}/train_z.npy'
            train_z = torch.tensor(np.load(embedding_save_path)).float()
            train_z = train_z[:, 1:, :]
            B, num_tokens, token_dim = train_z.size()
            
            syn_num, syn_cat = split_num_cat_target(syn_data, categories, d_numerical, pre_decoder, token_dim)
            
            syn_num = norm_fn.inverse_transform(syn_num)
            
            dn_syn = np.concatenate([syn_num, syn_cat], axis=1)
            dn_syn = pd.DataFrame(dn_syn, columns=feature_cols + label_cols)
            x_syn = dn_syn.iloc[:, :-1]
            y_syn = dn_syn.iloc[:, -1]
            synth_dir = os.path.join(exp_dir, f'synthesis/{random_seed}')
            if not os.path.exists(synth_dir):
                os.makedirs(synth_dir)
                
            x_syn.to_csv(os.path.join(synth_dir, 'x_syn.csv'))
            y_syn.to_csv(os.path.join(synth_dir, 'y_syn.csv'))
            print(f'seed: {random_seed}, xn_syn: {x_syn.shape}, y_syn: {y_syn.shape}')
        end_time = time.time()
        with open(os.path.join(exp_dir, 'time.txt'), 'a') as f:
            time_msg = f'\nsampling time: {end_time - start_time:.2f} seconds with {n_seeds} seeds'
            f.write(time_msg)
        print()
        
    if args.eval:
        # evaluate classifiers trained on synthetic data
        synth_dir_list = []
        for i in range(n_seeds):
            synth_dir = os.path.join(exp_dir, f'synthesis/{seed + i}')
            if os.path.exists(synth_dir):
                synth_dir_list.append(synth_dir)

        sst_col_names = data_desc['sst_col_names']
        metric = evaluate_syn_data(
            data_dir=os.path.join(data_config['path'], data_config['name']),
            exp_dir=exp_dir,
            synth_dir_list=synth_dir_list,
            sk_clf_lst=eval_config['sk_clf_choice'],
            sens_cols=sst_col_names,
        )

        with open(os.path.join(exp_dir, 'metric.json'), 'w') as f:
            json.dump(metric, f, indent=4)
            
        # print metric
        print_metric(metric)
        
if __name__ == '__main__':
    main()
